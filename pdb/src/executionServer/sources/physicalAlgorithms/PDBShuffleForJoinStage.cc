#include <PDBShuffleForJoinStage.h>
#include <PDBShuffleForJoinState.h>
#include <ComputePlan.h>
#include <ExJob.h>
#include <PDBStorageManagerBackend.h>
#include <GenericWork.h>

pdb::PDBShuffleForJoinStage::PDBShuffleForJoinStage(const pdb::PDBSinkPageSetSpec &sink,
                                                    const pdb::Vector<pdb::PDBSourceSpec> &sources,
                                                    const pdb::String &finalTupleSet,
                                                    const pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                                    const pdb::Vector<pdb::PDBSetObject> &setsToMaterialize,
                                                    const pdb::PDBSinkPageSetSpec &intermediate) :
                                                    PDBPhysicalAlgorithmStage(sink,
                                                                              sources,
                                                                              finalTupleSet,
                                                                              secondarySources,
                                                                              setsToMaterialize), intermediate(intermediate) {}

bool pdb::PDBShuffleForJoinStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                        const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                        const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                        const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBShuffleForJoinState>(state);

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  s->logicalPlan = plan.getPlan();

  /// 0. Make the intermediate page set

  // get the sink page set
  auto intermediatePageSet = storage->createAnonymousPageSet(intermediate.pageSetIdentifier);

  // did we manage to get a sink page set? if not the setup failed
  if (intermediatePageSet == nullptr) {
    return false;
  }

  /// 1. Init the shuffle queues

  s->pageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for(int i = 0; i < job->numberOfNodes; ++i) { s->pageQueues->emplace_back(std::make_shared<PDBPageQueue>()); }

  /// 2. Create the page set that contains the shuffled join side pages for this node

  // get the receive page set
  auto recvPageSet = storage->createFeedingAnonymousPageSet(std::make_pair(sink.pageSetIdentifier.first, sink.pageSetIdentifier.second),
                                                            job->numberOfProcessingThreads,
                                                            job->numberOfNodes);

  // make sure we can use them all at the same time
  recvPageSet->setUsagePolicy(PDBFeedingPageSetUsagePolicy::KEEP_AFTER_USED);

  // did we manage to get a page set where we receive this? if not the setup failed
  if(recvPageSet == nullptr) {
    return false;
  }

  /// 3. Create the self receiver to forward pages that are created on this node and the network senders to forward pages for the other nodes

  // get the buffer manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  int32_t currNode = job->thisNode;
  s->senders = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for(unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if(job->nodes[i]->port == job->nodes[currNode]->port && job->nodes[i]->address == job->nodes[currNode]->address) {

      // make the self receiver
      s->selfReceiver = std::make_shared<pdb::PDBPageSelfReceiver>(s->pageQueues->at(i), recvPageSet, myMgr);
    }
    else {

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(job->nodes[i]->address,
                                                           job->nodes[i]->port,
                                                           job->numberOfProcessingThreads,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           s->logger,
                                                           std::make_pair(sink.pageSetIdentifier.first, sink.pageSetIdentifier.second),
                                                           s->pageQueues->at(i));

      // setup the sender, if we fail return false
      if(!sender->setup()) {
        return false;
      }

      // make the sender
      s->senders->emplace_back(sender);
    }
  }

  /// 4. Initialize the sources

  // we put them here
  std::vector<PDBAbstractPageSetPtr> sourcePageSets;
  sourcePageSets.reserve(sources.size());

  // initialize them
  for(int i = 0; i < sources.size(); i++) {
    sourcePageSets.emplace_back(getSourcePageSet(storage, i));
  }

  /// 5. Initialize all the pipelines

  // get the number of worker threads from this server's config
  int32_t numWorkers = storage->getConfiguration()->numThreads;

  // check that we have at least one worker per primary source
  if(numWorkers < sources.size()) {
    return false;
  }

  /// 6. Figure out the source page set

  s->joinShufflePipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    /// 6.1. Figure out what source to use

    // figure out what pipeline
    auto pipelineSource = pipelineIndex % sources.size();

    // grab these thins from the source we need them
    bool swapLHSandRHS = sources[pipelineSource].swapLHSandRHS;
    const pdb::String &firstTupleSet = sources[pipelineSource].firstTupleSet;

    // get the source computation
    auto srcNode = s->logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

    // go grab the source page set
    PDBAbstractPageSetPtr sourcePageSet = sourcePageSets[pipelineSource];

    // did we manage to get a source page set? if not the setup failed
    if(sourcePageSet == nullptr) {
      return false;
    }

    /// 6.2. Figure out the parameters of the pipeline

    // figure out the join arguments
    auto joinArguments = getJoinArguments (storage);

    // if we could not create them we are out of here
    if(joinArguments == nullptr) {
      return false;
    }

    // get catalog client
    auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();

    // empty computations parameters
    std::map<ComputeInfoType, ComputeInfoPtr> params =  {{ComputeInfoType::PAGE_PROCESSOR, plan.getProcessorForJoin(finalTupleSet, job->numberOfNodes, job->numberOfProcessingThreads, *s->pageQueues, myMgr)},
                                                         {ComputeInfoType::JOIN_ARGS, joinArguments},
                                                         {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(swapLHSandRHS)},
                                                         {ComputeInfoType::SOURCE_SET_INFO, getSourceSetArg(catalogClient, pipelineSource)}};

    /// 6.3. Build the pipeline

    // build the join pipeline
    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       intermediatePageSet,
                                       params,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       20,
                                       pipelineIndex);

    // store the join pipeline
    s->joinShufflePipelines->push_back(pipeline);
  }

  return true;
}

bool pdb::PDBShuffleForJoinStage::run(const pdb::Handle<pdb::ExJob> &job,
                                      const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                      const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                      const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBShuffleForJoinState>(state);

  // success indicator
  atomic_bool success;
  success = true;

  /// 1. Run the self receiver,

  // create the buzzer
  atomic_int selfRecDone;
  selfRecDone = 0;
  PDBBuzzerPtr selfRefBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt = 1;
  });

  // run the work
  {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&selfRecDone, s](const PDBBuzzerPtr& callerBuzzer) {

      // run the receiver
      if(s->selfReceiver->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, selfRecDone);
      }
      else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, selfRecDone);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, selfRefBuzzer);
  }

  /// 2. Run the senders

  // create the buzzer
  atomic_int sendersDone;
  sendersDone = 0;
  PDBBuzzerPtr sendersBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  // go through each sender and run them
  for(auto &sender : *s->senders) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendersDone, sender, this](const PDBBuzzerPtr& callerBuzzer) {

      // run the sender
      if(sender->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, sendersDone);
      }
      else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, sendersDone);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, sendersBuzzer);
  }

  /// 3. Run the join pipelines

  // create the buzzer
  atomic_int joinCounter;
  joinCounter = 0;
  PDBBuzzerPtr joinBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // here we get a worker per pipeline and run all the shuffle join side pipelines.
  for (int workerID = 0; workerID < s->joinShufflePipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&joinCounter, &success, workerID, s](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        (*s->joinShufflePipelines)[workerID]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, joinCounter);
    });

    // run the work
    worker->execute(myWork, joinBuzzer);
  }

  // wait until all the shuffle join side pipelines have completed
  while (joinCounter < s->joinShufflePipelines->size()) {
    joinBuzzer->wait();
  }

  // ok they have finished now push a null page to each of the preagg queues
  for(auto &queue : *s->pageQueues) { queue->enqueue(nullptr); }

  // wait while we are running the receiver
  while(selfRecDone == 0) {
    selfRefBuzzer->wait();
  }

  // wait while we are running the senders
  while(sendersDone < s->senders->size()) {
    sendersBuzzer->wait();
  }

  return true;
}

void pdb::PDBShuffleForJoinStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBShuffleForJoinState>(state);

  // invalidate everything
  s->pageQueues = nullptr;
  s->joinShufflePipelines = nullptr;
  s->logger = nullptr;
  s->selfReceiver = nullptr;
  s->senders = nullptr;
  s->logicalPlan = nullptr;
}