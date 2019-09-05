//
// Created by dimitrije on 5/7/19.
//

#include <physicalAlgorithms/PDBBroadcastForJoinAlgorithm.h>

pdb::PDBBroadcastForJoinAlgorithm::PDBBroadcastForJoinAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                                                                const AtomicComputationPtr &finalAtomicComputation,
                                                                const pdb::Handle<pdb::PDBSinkPageSetSpec> &hashedToSend,
                                                                const pdb::Handle<pdb::PDBSourcePageSetSpec> &hashedToRecv,
                                                                const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                                                                const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                                                const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize):
                                                                      PDBPhysicalAlgorithm(primarySource,
                                                                                           finalAtomicComputation,
                                                                                           sink,
                                                                                           secondarySources,
                                                                                           setsToMaterialize),
                                                                      hashedToSend(hashedToSend),
                                                                      hashedToRecv(hashedToRecv) {
}

pdb::PDBPhysicalAlgorithmType pdb::PDBBroadcastForJoinAlgorithm::getAlgorithmType() {
  return BroadcastForJoin;
}

bool pdb::PDBBroadcastForJoinAlgorithm::setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                              Handle<pdb::ExJob> &job,
                                              const std::string &error) {

  // init the plan
  ComputePlan plan(job->tcap, *job->computations);
  logicalPlan = plan.getPlan();

  // get the manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // init the logger
  logger = make_shared<PDBLogger>("BroadcastJoinPipeAlgorithm" + std::to_string(job->computationID));

  /// 0. Figure out the sink tuple set for the prebroadcastjoin.

  // get the sink page set
  auto intermediatePageSet = storage->createAnonymousPageSet(hashedToSend->pageSetIdentifier);

  // did we manage to get a sink page set? if not the setup failed
  if (intermediatePageSet == nullptr) {
    return false;
  }

  /// 1. Init the prebroadcastjoin queues

  pageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for (int i = 0; i < job->numberOfNodes; ++i) { pageQueues->emplace_back(std::make_shared<PDBPageQueue>()); }

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
    auto srcNode = logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

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
    std::map<ComputeInfoType, ComputeInfoPtr> params = {{ComputeInfoType::PAGE_PROCESSOR,std::make_shared<BroadcastJoinProcessor>(job->numberOfNodes,job->numberOfProcessingThreads,*pageQueues,myMgr)},
                                                        {ComputeInfoType::JOIN_ARGS, joinArguments},
                                                        {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(swapLHSandRHS)},
                                                        {ComputeInfoType::SOURCE_SET_INFO, getSourceSetArg(catalogClient, pipelineIndex)}};

    /// 3.2. create the prebroadcastjoin pipelines

    // fill uo the vector for each thread
    prebroadcastjoinPipelines = std::make_shared<std::vector<PipelinePtr>>();


    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       intermediatePageSet,
                                       params,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       20,
                                       pipelineIndex);

    prebroadcastjoinPipelines->push_back(pipeline);
  }

  // get the sink page set
  auto sinkPageSet = storage->createAnonymousPageSet(std::make_pair(sink->pageSetIdentifier.first, sink->pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if (sinkPageSet == nullptr) {
    return false;
  }

  // set it to concurrent since each thread needs to use the same pages
  sinkPageSet->setAccessOrder(PDBAnonymousPageSetAccessPattern::CONCURRENT);

  /// 4. Create the page set that contains the prebroadcastjoin pages for this node

  // get the receive page set
  auto recvPageSet = storage->createFeedingAnonymousPageSet(std::make_pair(hashedToRecv->pageSetIdentifier.first,
                                                                           hashedToRecv->pageSetIdentifier.second),
                                                            job->numberOfProcessingThreads,
                                                            job->numberOfNodes);

  // did we manage to get a page set where we receive this? if not the setup failed
  if (recvPageSet == nullptr) {
    return false;
  }

  /// 5. Create the self receiver to forward pages that are created on this node and the network senders to forward pages for the other nodes

  senders = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for (unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if (job->nodes[i]->port == job->thisNode->port && job->nodes[i]->address == job->thisNode->address) {

      // make the self receiver
      selfReceiver = std::make_shared<pdb::PDBPageSelfReceiver>(pageQueues->at(i), recvPageSet, myMgr);
    } else {

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(job->nodes[i]->address,
                                                           job->nodes[i]->port,
                                                           job->numberOfProcessingThreads,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           logger,
                                                           std::make_pair(hashedToRecv->pageSetIdentifier.first,
                                                                          hashedToRecv->pageSetIdentifier.second),
                                                           pageQueues->at(i));

      // setup the sender, if we fail return false
      if (!sender->setup()) {
        return false;
      }

      // make the sender
      senders->emplace_back(sender);
    }
  }

  /// 6. Create the broadcastjoin pipeline

  broadcastjoinPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t workerID = 0; workerID < job->numberOfProcessingThreads; ++workerID) {

    // build the broadcastjoin pipeline
    auto joinbroadcastPipeline = plan.buildBroadcastJoinPipeline(finalTupleSet,
                                                                 recvPageSet,
                                                                 sinkPageSet,
                                                                 job->numberOfProcessingThreads,
                                                                 job->numberOfNodes,
                                                                 workerID);

    // store the broadcastjoin pipeline
    broadcastjoinPipelines->push_back(joinbroadcastPipeline);
  }

  return true;
}

bool pdb::PDBBroadcastForJoinAlgorithm::run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

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
  for (int workerID = 0; workerID < broadcastjoinPipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&joinCounter, workerID, this](PDBBuzzerPtr callerBuzzer) {

      // run the pipeline
      (*broadcastjoinPipelines)[workerID]->run();

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
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&selfRecDone, this](PDBBuzzerPtr callerBuzzer) {

      // run the receiver
      if (selfReceiver->run()) {

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
  sendersDone = senders->size();
  PDBBuzzerPtr sendersBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  // go through each sender and run them
  for (auto &sender : *senders) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendersDone, sender, this](PDBBuzzerPtr callerBuzzer) {

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
  for (int workerID = 0; workerID < prebroadcastjoinPipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&prejoinCounter, workerID, this](PDBBuzzerPtr callerBuzzer) {

      // run the pipeline
      (*prebroadcastjoinPipelines)[workerID]->run();

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, prejoinCounter);
    });

    // run the work
    worker->execute(myWork, prejoinBuzzer);
  }

  /// 5. Do the waiting

  // wait until all the prebroadcastjoinpipelines have completed
  while (prejoinCounter < prebroadcastjoinPipelines->size()) {
    prejoinBuzzer->wait();
  }

  // ok they have finished now push a null page to each of the queues
  for (auto &queue : *pageQueues) { queue->enqueue(nullptr); }

  // wait while we are running the receiver
  while (selfRecDone == 0) {
    selfRefBuzzer->wait();
  }

  // wait while we are running the senders
  while (sendersDone < senders->size()) {
    sendersBuzzer->wait();
  }

  // wait until all the broadcastjoin pipelines have completed
  while (joinCounter < broadcastjoinPipelines->size()) {
    joinBuzzer->wait();
  }

  return true;
}

void pdb::PDBBroadcastForJoinAlgorithm::cleanup() {
  hashedToSend = nullptr;
  hashedToRecv = nullptr;
  selfReceiver = nullptr;
  senders = nullptr;
  logger = nullptr;
  prebroadcastjoinPipelines = nullptr;
  broadcastjoinPipelines = nullptr;
  pageQueues = nullptr;
}