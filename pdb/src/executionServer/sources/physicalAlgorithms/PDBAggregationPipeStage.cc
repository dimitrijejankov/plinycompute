#include <PDBAggregationPipeStage.h>
#include <PDBAggregationPipeState.h>
#include <PreaggregationPageProcessor.h>
#include <ComputePlan.h>
#include <PDBCatalogClient.h>
#include <ExJob.h>
#include <GenericWork.h>

pdb::PDBAggregationPipeStage::PDBAggregationPipeStage(const pdb::PDBSinkPageSetSpec &sink,
                                                      const pdb::Vector<pdb::PDBSourceSpec> &sources,
                                                      const pdb::String &final_tuple_set,
                                                      const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> &secondary_sources,
                                                      const pdb::Vector<pdb::PDBSetObject> &sets_to_materialize,
                                                      const pdb::PDBSinkPageSetSpec &hashed_to_send,
                                                      const pdb::PDBSourcePageSetSpec &hashed_to_recv)
    : PDBPhysicalAlgorithmStage(sink, sources, final_tuple_set, secondary_sources, sets_to_materialize),
      hashedToSend(hashed_to_send),
      hashedToRecv(hashed_to_recv) {}

bool pdb::PDBAggregationPipeStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                         const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                         const std::shared_ptr<pdb::PDBStorageManager> &storage,
                                         const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBAggregationPipeState>(state);

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  s->logicalPlan = plan.getPlan();

  // get the buffer manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();


  /// 1. Figure out the sink tuple set for the preaggregation (this will provide empty pages to the pipeline but we will
  /// discard them since they will be processed by the PreaggregationPageProcessor and they won't stay around).

  // get the sink page set
  auto intermediatePageSet = storage->createAnonymousPageSet(hashedToSend.pageSetIdentifier);

  // did we manage to get a sink page set? if not the setup failed
  if (intermediatePageSet == nullptr) {
    return false;
  }

  /// 2. Init the preaggregation queues

  s->pageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for(int i = 0; i < job->numberOfNodes; ++i) { s->pageQueues->emplace_back(std::make_shared<PDBPageQueue>()); }


  /// 3. Initialize the sources

  // we put them here
  std::vector<PDBAbstractPageSetPtr> sourcePageSets;
  sourcePageSets.reserve(sources.size());

  // initialize them
  for(int i = 0; i < sources.size(); i++) {
    sourcePageSets.emplace_back(getSourcePageSet(storage, i));
  }

  /// 4. Initialize all the pipelines

  // get the number of worker threads from this server's config
  int32_t numWorkers = storage->getConfiguration()->numThreads;

  // check that we have at least one worker per primary source
  if(numWorkers < sources.size()) {
    return false;
  }

  // fill uo the vector for each thread
  s->preaggregationPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    /// 4.1. Figure out the source page set

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
    if (sourcePageSet == nullptr) {
      return false;
    }

    /// 4.2. Figure out the parameters of the pipeline

    // figure out the join arguments
    auto joinArguments = getJoinArguments (storage);

    // if we could not create them we are out of here
    if(joinArguments == nullptr) {
      return false;
    }

    // get catalog client
    auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();

    // initialize the parameters
    std::map<ComputeInfoType, ComputeInfoPtr> params = { { ComputeInfoType::PAGE_PROCESSOR,  std::make_shared<PreaggregationPageProcessor>(job->numberOfNodes,
                                                                                                                                           job->numberOfProcessingThreads,
                                                                                                                                           *s->pageQueues,
                                                                                                                                           myMgr) },
                                                         { ComputeInfoType::JOIN_ARGS, joinArguments },
                                                         { ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(swapLHSandRHS) },
                                                         { ComputeInfoType::SOURCE_SET_INFO, getSourceSetArg(catalogClient, pipelineSource)}} ;

    /// 4.3. Build the pipeline

    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       intermediatePageSet,
                                       params,
                                       job->thisNode,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       pipelineIndex);

    s->preaggregationPipelines->push_back(pipeline);
  }

  /// 5. Create the sink

  // get the sink page set
  auto sinkPageSet = storage->createAnonymousPageSet(std::make_pair(sink.pageSetIdentifier.first, sink.pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if(sinkPageSet == nullptr) {
    return false;
  }

  /// 6. Create the page set that contains the preaggregated pages for this node

  // get the receive page set
  auto recvPageSet = storage->createFeedingAnonymousPageSet(std::make_pair(hashedToRecv.pageSetIdentifier.first, hashedToRecv.pageSetIdentifier.second),
                                                            job->numberOfProcessingThreads,
                                                            job->numberOfNodes);

  // did we manage to get a page set where we receive this? if not the setup failed
  if(recvPageSet == nullptr) {
    return false;
  }

  /// 7. Create the self receiver to forward pages that are created on this node and the network senders to forward pages for the other nodes
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
      auto sender = std::make_shared<PDBPageNetworkSender>(storage->getConMgr(),
                                                           job->nodes[i]->nodeID,
                                                           job->numberOfProcessingThreads,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           s->logger,
                                                           std::make_pair(hashedToRecv.pageSetIdentifier.first, hashedToRecv.pageSetIdentifier.second),
                                                           s->pageQueues->at(i));

      // setup the sender, if we fail return false
      if(!sender->setup()) {
        return false;
      }

      // make the sender
      s->senders->emplace_back(sender);
    }
  }

  /// 8. Create the aggregation pipeline

  s->aggregationPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t workerID = 0; workerID < job->numberOfProcessingThreads; ++workerID) {

    // build the aggregation pipeline
    auto aggPipeline = plan.buildAggregationPipeline(finalTupleSet, myMgr->getWorkerQueue(), recvPageSet, sinkPageSet, workerID);

    // store the aggregation pipeline
    s->aggregationPipelines->push_back(aggPipeline);
  }

  return true;
}

bool pdb::PDBAggregationPipeStage::run(const pdb::Handle<pdb::ExJob> &job,
                                       const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                       const std::shared_ptr<pdb::PDBStorageManager> &storage,
                                       const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBAggregationPipeState>(state);

  // success indicator
  atomic_bool success;
  success = true;

  /// 1. Run the aggregation pipeline, this runs after the preaggregation pipeline, but is started first.

  // create the buzzer
  atomic_int aggCounter;
  aggCounter = 0;
  PDBBuzzerPtr aggBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // here we get a worker per pipeline and run all the preaggregation Pipelines.
  for (int workerID = 0; workerID < s->aggregationPipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&aggCounter, &success, workerID, &s](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        (*s->aggregationPipelines)[workerID]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, aggCounter);
    });

    // run the work
    worker->execute(myWork, aggBuzzer);
  }

  /// 2. Run the self receiver so it can server pages to the aggregation pipeline

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

  /// 4. Run the preaggregation, this step comes before the aggregation step

  // create the buzzer
  atomic_int preaggCounter;
  preaggCounter = 0;
  PDBBuzzerPtr preaggBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // here we get a worker per pipeline and run all the preaggregationPipelines.
  for (int workerID = 0; workerID < s->preaggregationPipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&preaggCounter, &success, workerID, &s](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        (*s->preaggregationPipelines)[workerID]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, preaggCounter);
    });

    // run the work
    worker->execute(myWork, preaggBuzzer);
  }

  /// 5. Do the waiting

  // wait until all the preaggregationPipelines have completed
  while (preaggCounter < s->preaggregationPipelines->size()) {
    preaggBuzzer->wait();
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

  // wait until all the aggregation pipelines have completed
  while (aggCounter < s->aggregationPipelines->size()) {
    aggBuzzer->wait();
  }

  /// 6. Should we materialize

  // should we materialize this to a set?
  for(int j = 0; j < setsToMaterialize.size(); ++j) {

    // get the page set
    auto sinkPageSet = storage->getPageSet(std::make_pair(sink.pageSetIdentifier.first, sink.pageSetIdentifier.second));

    // if the thing does not exist finish!
    if(sinkPageSet == nullptr) {
      success = false;
      break;
    }

    // materialize the page set
    sinkPageSet->resetPageSet();
    success = storage->materializePageSet(sinkPageSet, std::make_pair<std::string, std::string>(setsToMaterialize[j].database, setsToMaterialize[j].set)) && success;
  }

  return true;
}

void pdb::PDBAggregationPipeStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManager> &storage) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBAggregationPipeState>(state);

  // invalidate all the ptrs this should destroy everything
  s->selfReceiver = nullptr;
  s->senders = nullptr;
  s->logger = nullptr;
  s->preaggregationPipelines = nullptr;
  s->aggregationPipelines = nullptr;
  s->pageQueues = nullptr;
}