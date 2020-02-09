#include <PDBBroadcastForJoinStage.h>
#include <PDBBroadcastForJoinState.h>
#include <BroadcastJoinProcessor.h>
#include <ComputePlan.h>
#include <GenericWork.h>
#include <ExJob.h>

pdb::PDBBroadcastForJoinStage::PDBBroadcastForJoinStage(const pdb::PDBSinkPageSetSpec &sink,
                                                        const pdb::Vector<pdb::PDBSourceSpec> &sources,
                                                        const pdb::String &final_tuple_set,
                                                        const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> &secondary_sources,
                                                        const pdb::Vector<pdb::PDBSetObject> &sets_to_materialize,
                                                        pdb::PDBSinkPageSetSpec &hashed_to_send,
                                                        pdb::PDBSourcePageSetSpec &hashed_to_recv)
    : PDBPhysicalAlgorithmStage(sink, sources, final_tuple_set, secondary_sources, sets_to_materialize),
      hashedToSend(hashed_to_send),
      hashedToRecv(hashed_to_recv) {}

bool pdb::PDBBroadcastForJoinStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                          const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                          const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                          const std::string &error) {


  // cast the state
  auto s = dynamic_pointer_cast<PDBBroadcastForJoinState>(state);

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  s->logicalPlan = plan.getPlan();

  // get the manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  /// 0. Figure out the sink tuple set for the prebroadcastjoin.

  // get the sink page set
  auto intermediatePageSet = storage->createAnonymousPageSet(hashedToSend.pageSetIdentifier);

  // did we manage to get a sink page set? if not the setup failed
  if (intermediatePageSet == nullptr) {
    return false;
  }

  /// 1. Init the prebroadcastjoin queues

  s->pageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for (int i = 0; i < job->numberOfNodes; ++i) { s->pageQueues->emplace_back(std::make_shared<PDBPageQueue>()); }

  /// 2. Initialize the sources

  // we put them here
  std::vector<PDBAbstractPageSetPtr> sourcePageSets;
  sourcePageSets.reserve(sources.size());

  // initialize them
  for(int i = 0; i < sources.size(); i++) {
    sourcePageSets.emplace_back(getSourcePageSet(storage, i));
  }

  /// 3. Initialize all the pipelines

  // get the number of worker threads from this server's config
  int32_t numWorkers = storage->getConfiguration()->numThreads;

  // check that we have at least one worker per primary source
  if(numWorkers < sources.size()) {
    return false;
  }

  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    // figure out what pipeline
    auto pipelineSource = pipelineIndex % sources.size();

    // grab these thins from the source we need them
    bool swapLHSandRHS = sources[pipelineSource].swapLHSandRHS;
    const pdb::String &firstTupleSet = sources[pipelineSource].firstTupleSet;

    // get the source computation
    auto srcNode = s->logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

    // if this is a scan set get the page set from a real set
    PDBAbstractPageSetPtr sourcePageSet = getSourcePageSet(storage, pipelineIndex);

    // did we manage to get a source page set? if not the setup failed
    if (sourcePageSet == nullptr) {
      return false;
    }

    /// 3.1. Init the prebroadcastjoin pipeline parameters

    // figure out the join arguments
    auto joinArguments = getJoinArguments(storage);

    // if we could not create them we are out of here
    if (joinArguments == nullptr) {
      return false;
    }

    // get catalog client
    auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();

    // set the parameters
    std::map<ComputeInfoType, ComputeInfoPtr> params = {{ComputeInfoType::PAGE_PROCESSOR,std::make_shared<BroadcastJoinProcessor>(job->numberOfNodes,job->numberOfProcessingThreads,*s->pageQueues, myMgr)},
                                                        {ComputeInfoType::JOIN_ARGS, joinArguments},
                                                        {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(swapLHSandRHS)},
                                                        {ComputeInfoType::SOURCE_SET_INFO, getSourceSetArg(catalogClient, pipelineIndex)}};

    /// 3.2. create the prebroadcastjoin pipelines

    // fill uo the vector for each thread
    s->prebroadcastjoinPipelines = std::make_shared<std::vector<PipelinePtr>>();


    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       intermediatePageSet,
                                       params,
                                       job->thisNode,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       pipelineIndex);

    s->prebroadcastjoinPipelines->push_back(pipeline);
  }

  // get the sink page set
  auto sinkPageSet = storage->createAnonymousPageSet(std::make_pair(sink.pageSetIdentifier.first, sink.pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if (sinkPageSet == nullptr) {
    return false;
  }

  // set it to concurrent since each thread needs to use the same pages
  sinkPageSet->setAccessOrder(PDBAnonymousPageSetAccessPattern::CONCURRENT);

  /// 4. Create the page set that contains the prebroadcastjoin pages for this node

  // get the receive page set
  auto recvPageSet = storage->createFeedingAnonymousPageSet(std::make_pair(hashedToRecv.pageSetIdentifier.first,
                                                                           hashedToRecv.pageSetIdentifier.second),
                                                            job->numberOfProcessingThreads,
                                                            job->numberOfNodes);

  // did we manage to get a page set where we receive this? if not the setup failed
  if (recvPageSet == nullptr) {
    return false;
  }

  /// 5. Create the self receiver to forward pages that are created on this node and the network senders to forward pages for the other nodes

  int32_t currNode = job->thisNode;
  s->senders = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for (unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if (job->nodes[i]->port == job->nodes[currNode]->port && job->nodes[i]->address == job->nodes[currNode]->address) {

      // make the self receiver
      s->selfReceiver = std::make_shared<pdb::PDBPageSelfReceiver>(s->pageQueues->at(i), recvPageSet, myMgr);
    } else {

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(job->nodes[i]->address,
                                                           job->nodes[i]->port,
                                                           job->numberOfProcessingThreads,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           s->logger,
                                                           std::make_pair(hashedToRecv.pageSetIdentifier.first,
                                                                          hashedToRecv.pageSetIdentifier.second),
                                                           s->pageQueues->at(i));

      // setup the sender, if we fail return false
      if (!sender->setup()) {
        return false;
      }

      // make the sender
      s->senders->emplace_back(sender);
    }
  }

  /// 6. Create the broadcastjoin pipeline

  s->broadcastjoinPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t workerID = 0; workerID < job->numberOfProcessingThreads; ++workerID) {

    // build the broadcastjoin pipeline
    auto joinbroadcastPipeline = plan.buildBroadcastJoinPipeline(finalTupleSet,
                                                                 recvPageSet,
                                                                 sinkPageSet,
                                                                 job->numberOfProcessingThreads,
                                                                 job->numberOfNodes,
                                                                 workerID);

    // store the broadcastjoin pipeline
    s->broadcastjoinPipelines->push_back(joinbroadcastPipeline);
  }

  return true;
}

bool pdb::PDBBroadcastForJoinStage::run(const pdb::Handle<pdb::ExJob> &job,
                                        const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                        const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                        const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBBroadcastForJoinState>(state);

  // success indicator
  atomic_bool success;
  success = true;

  /// 1. Run the broadcastjoin (merge) pipeline, this runs after the prebroadcastjoin pipelines, but is started first.

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

  // here we get a worker per pipeline and run all the Broadcastjoin Pipelines.
  for (int workerID = 0; workerID < s->broadcastjoinPipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&joinCounter, &success, workerID, &s](const PDBBuzzerPtr &callerBuzzer) {

      try {

        // run the pipeline
        (*s->broadcastjoinPipelines)[workerID]->run();
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

  /// 2. Run the self receiver so it can server pages to the broadcastjoin pipeline

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
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&selfRecDone, &s](const PDBBuzzerPtr& callerBuzzer) {

      // run the receiver
      if (s->selfReceiver->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, selfRecDone);
      } else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, selfRecDone);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, selfRefBuzzer);
  }

  /// 3. Run the senders

  // create the buzzer
  atomic_int sendersDone;
  sendersDone = s->senders->size();
  PDBBuzzerPtr sendersBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  // go through each sender and run them
  for (auto &sender : *s->senders) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendersDone, sender](const PDBBuzzerPtr& callerBuzzer) {

      // run the sender
      if (sender->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, sendersDone);
      } else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, sendersDone);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, sendersBuzzer);
  }

  /// 4. Run the prebroadcastjoin, this step comes before the broadcastjoin (merge) step

  // create the buzzer
  atomic_int prejoinCounter;
  prejoinCounter = 0;
  PDBBuzzerPtr prejoinBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // here we get a worker per pipeline and run all the prebroadcastjoinPipelines.
  for (int workerID = 0; workerID < s->prebroadcastjoinPipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&prejoinCounter, &success, workerID, &s](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        (*s->prebroadcastjoinPipelines)[workerID]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, prejoinCounter);
    });

    // run the work
    worker->execute(myWork, prejoinBuzzer);
  }

  /// 5. Do the waiting

  // wait until all the pre-broadcast join pipelines have completed
  while (prejoinCounter < s->prebroadcastjoinPipelines->size()) {
    prejoinBuzzer->wait();
  }

  // ok they have finished now push a null page to each of the queues
  for (auto &queue : *s->pageQueues) { queue->enqueue(nullptr); }

  // wait while we are running the receiver
  while (selfRecDone == 0) {
    selfRefBuzzer->wait();
  }

  // wait while we are running the senders
  while (sendersDone < s->senders->size()) {
    sendersBuzzer->wait();
  }

  // wait until all the broadcastjoin pipelines have completed
  while (joinCounter < s->broadcastjoinPipelines->size()) {
    joinBuzzer->wait();
  }

  return true;
}

void pdb::PDBBroadcastForJoinStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) {


  // cast the state
  auto s = dynamic_pointer_cast<PDBBroadcastForJoinState>(state);

  // set stuff to null to invalidate
  s->selfReceiver = nullptr;
  s->senders = nullptr;
  s->logger = nullptr;
  s->prebroadcastjoinPipelines = nullptr;
  s->broadcastjoinPipelines = nullptr;
  s->pageQueues = nullptr;
}