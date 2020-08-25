#include <AtomicComputationClasses.h>
#include <PDBPhysicalAlgorithm.h>
#include "PDBJoinAggregationComputationStage.h"
#include "PDBJoinAggregationState.h"
#include "ComputePlan.h"
#include "LogicalPlanTransformer.h"
#include "DismissProcessor.h"
#include "PreaggregationPageProcessor.h"
#include "GenericWork.h"
#include "ExJob.h"
#include "PDBCatalogClient.h"

pdb::PDBJoinAggregationComputationStage::PDBJoinAggregationComputationStage(const pdb::PDBSinkPageSetSpec &sink,
                                                                            const pdb::PDBSinkPageSetSpec &preaggIntermediate,
                                                                            const pdb::Vector<pdb::PDBSourceSpec> &sources,
                                                                            const pdb::String &final_tuple_set,
                                                                            const pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>> &secondary_sources,
                                                                            const pdb::Vector<pdb::PDBSetObject> &sets_to_materialize,
                                                                            const pdb::String &join_tuple_set,
                                                                            const pdb::PDBSourcePageSetSpec &left_join_source,
                                                                            const pdb::PDBSourcePageSetSpec &right_join_source,
                                                                            const pdb::PDBSinkPageSetSpec &intermediate_sink,
                                                                            const pdb::Vector<PDBSourceSpec> &right_sources)
    : PDBPhysicalAlgorithmStage(sink, sources, final_tuple_set, secondary_sources, sets_to_materialize),
      joinTupleSet(join_tuple_set),
      leftJoinSource(left_join_source),
      rightJoinSource(right_join_source),
      intermediateSink(intermediate_sink),
      preaggIntermediate(preaggIntermediate),
      rightSources(right_sources) {}

bool pdb::PDBJoinAggregationComputationStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                                    const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                                    const std::shared_ptr<pdb::PDBStorageManager> &storage,
                                                    const std::string &error) {


  // cast the state
  auto s = dynamic_pointer_cast<PDBJoinAggregationState>(state);

  // get catalog client
  auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  s->logicalPlan = plan.getPlan();

  // make the transformer
  auto transformer = std::make_shared<LogicalPlanTransformer>(s->logicalPlan);
  transformer->addTransformation(std::make_shared<DropToKeyExtractionTransformation>(joinTupleSet));

  // apply all the transformations
  s->logicalPlan = transformer->applyTransformations();

  // get the join comp
  auto joinAtomicComp =
      dynamic_pointer_cast<ApplyJoin>(s->logicalPlan->getComputations().getProducingAtomicComputation(joinTupleSet));

  std::cout << "Exec plan" << *s->logicalPlan << '\n';

  /// 10. Make outgoing connections to other nodes

  // make the object
  UseTemporaryAllocationBlock tmp{1024};
  pdb::Handle<SerConnectToRequest> connectionID = pdb::makeObject<SerConnectToRequest>(job->computationID,
                                                                                       job->jobID,
                                                                                       job->thisNode,
                                                                                       PDBJoinAggregationState::LEFT_JOIN_SIDE_TASK);

  // init the vector for the left sides
  s->leftJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // connect to the node
    s->leftJoinSideCommunicatorsOut->push_back(myMgr->connectTo(job->nodes[n]->nodeID, connectionID));
  }

  // init the vector for the right sides
  connectionID->taskID = PDBJoinAggregationState::RIGHT_JOIN_SIDE_TASK;
  s->rightJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // connect to the node
    s->rightJoinSideCommunicatorsOut->push_back(myMgr->connectTo(job->nodes[n]->nodeID, connectionID));
  }

  /// 11. Get the incoming connections to this node.

  // wait for left side connections
  connectionID->taskID = PDBJoinAggregationState::LEFT_JOIN_SIDE_TASK;
  s->leftJoinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // wait for the connection
    s->leftJoinSideCommunicatorsIn->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if (s->leftJoinSideCommunicatorsIn->back()->isSocketClosed()) {

      // log the error
      s->logger->error("Socket for the left side is closed");

      return false;
    }
  }

  // wait for the right side connections
  connectionID->taskID = PDBJoinAggregationState::RIGHT_JOIN_SIDE_TASK;
  s->rightJoinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // wait for the connection
    s->rightJoinSideCommunicatorsIn->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if (s->rightJoinSideCommunicatorsIn->back()->isSocketClosed()) {

      // log the error
      s->logger->error("Socket for the right side is closed");

      return false;
    }
  }

  /// 12. Setup the join side senders

  // get the join computation
  auto joinComp = getJoinComp(s->logicalPlan);

  // setup the left senders
  s->leftJoinSideSenders = std::make_shared<std::vector<JoinAggSideSenderPtr>>();

  // init the senders
  for (auto &comm : *s->leftJoinSideCommunicatorsIn) {

    // init the right senders
    s->leftJoinSideSenders->emplace_back(joinComp->getJoinAggSender(joinAtomicComp->getProjection(),
                                                                    s->logicalPlan,
                                                                    myMgr->getPage(),
                                                                    comm));
  }

  // setup the right senders
  s->rightJoinSideSenders = std::make_shared<std::vector<JoinAggSideSenderPtr>>();

  // init the senders
  for (auto &comm : *s->rightJoinSideCommunicatorsIn) {

    // init the right senders
    s->rightJoinSideSenders->emplace_back(joinComp->getJoinAggSender(joinAtomicComp->getRightProjection(),
                                                                     s->logicalPlan,
                                                                     myMgr->getPage(),
                                                                     comm));
  }

  /// 13. Setup the join map creators

  // create the join tuple emitter
  s->emitter = std::make_shared<JoinAggTupleEmitter>(s->planPage, job->numberOfProcessingThreads, job->thisNode);

  s->leftJoinMapCreators = std::make_shared<std::vector<JoinMapCreatorPtr>>();

  // init the join side creators
  s->leftShuffledPageSet = storage->createRandomAccessPageSet(leftJoinSource.pageSetIdentifier);
  for (auto &comm : *s->leftJoinSideCommunicatorsOut) {

    // make the creators
    s->leftJoinMapCreators->emplace_back(joinComp->getJoinMapCreator(joinAtomicComp->getProjection(),
                                                                     s->logicalPlan,
                                                                     s->leftShuffledPageSet,
                                                                     comm,
                                                                     s->emitter,
                                                                     true,
                                                                     s->logger));
  }

  // setup the right one
  s->rightJoinMapCreators = std::make_shared<std::vector<JoinMapCreatorPtr>>();

  // init the join side creators
  s->rightShuffledPageSet = storage->createRandomAccessPageSet(rightJoinSource.pageSetIdentifier);
  for (auto &comm : *s->rightJoinSideCommunicatorsOut) {

    // make the creators
    s->rightJoinMapCreators->emplace_back(joinComp->getJoinMapCreator(joinAtomicComp->getProjection(),
                                                                      s->logicalPlan,
                                                                      s->rightShuffledPageSet,
                                                                      comm,
                                                                      s->emitter,
                                                                      false,
                                                                      s->logger));
  }

  /// 14. the left and right side of the join

  // the join key pipelines
  s->joinPipelines = make_shared<std::vector<PipelinePtr>>();

  // get the sink page set
  s->intermediatePageSet = storage->createAnonymousPageSet(std::make_pair(intermediateSink.pageSetIdentifier.first,
                                                                          intermediateSink.pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if (s->intermediatePageSet == nullptr) {
    return false;
  }

  /// 14.1 Initialize the left sources and the sink

  // we put them here
  std::vector<PDBAbstractPageSetPtr> leftSourcePageSets;
  leftSourcePageSets.reserve(sources.size());

  // initialize them
  for (int i = 0; i < sources.size(); i++) {
    leftSourcePageSets.emplace_back(getSourcePageSet(storage, i));
  }

  /// 14.2. For each node initialize left pipelines

  // how many threads we are going to use per join side
  int threads_per = std::max(job->numberOfProcessingThreads / 2, sources.size());

  // initialize all the pipelines
  std::map<ComputeInfoType, ComputeInfoPtr> params;
  for (uint64_t pipelineIndex = 0; pipelineIndex < threads_per; ++pipelineIndex) {

    /// 14.2.1 Figure out what source to use

    // figure out what pipeline
    auto pipelineSource = pipelineIndex % sources.size();

    // grab these thins from the source we need them
    bool swapLHSandRHS = sources[pipelineSource].swapLHSandRHS;
    const pdb::String &firstTupleSet = sources[pipelineSource].firstTupleSet;

    // get the source computation
    auto srcNode = s->logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

    // go grab the source page set
    PDBAbstractPageSetPtr sourcePageSet = leftSourcePageSets[pipelineSource];

    // did we manage to get a source page set? if not the setup failed
    if (sourcePageSet == nullptr) {
      return false;
    }

    /// 14.2.2 Figure out the parameters of the pipeline

    // figure out the join arguments
    auto joinArguments = getJoinArguments(storage);

    // if we could not create them we are out of here
    if (joinArguments == nullptr) {
      return false;
    }

    // mark that this is a join agg side
    joinArguments->isJoinAggSide = true;

    // empty computations parameters
    params = {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<DismissProcessor>()},
              {ComputeInfoType::JOIN_ARGS, joinArguments},
              {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(swapLHSandRHS)},
              {ComputeInfoType::JOIN_AGG_SIDE_ARGS,
               std::make_shared<JoinAggSideArg>(s->leftKeyPage, s->leftJoinSideSenders, s->planPage, true)},
              {ComputeInfoType::SOURCE_SET_INFO, getSourceSetArg(catalogClient, pipelineSource)}};

    /// 14.2.3 Build the pipeline

    // build the join pipeline
    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       joinAtomicComp->getInputName(),     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       s->intermediatePageSet,
                                       params,
                                       job->thisNode,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       pipelineIndex);

    // store the join pipeline
    s->joinPipelines->push_back(pipeline);
  }

  /// 14.3 Initialize the right sources

  // we put them here
  std::vector<PDBAbstractPageSetPtr> rightSourcePageSets;
  rightSourcePageSets.reserve(sources.size());

  // initialize them
  for (int i = 0; i < rightSources.size(); i++) {
    rightSourcePageSets.emplace_back(getRightSourcePageSet(storage, i));
  }

  /// 14.4. For each node initialize left pipelines

  // initialize all the pipelines
  for (uint64_t pipelineIndex = 0; pipelineIndex < threads_per; ++pipelineIndex) {

    /// 14.4.1 Figure out what source to use

    // figure out what pipeline
    auto pipelineSource = pipelineIndex % rightSources.size();

    // grab these thins from the source we need them
    bool swapLHSandRHS = rightSources[pipelineSource].swapLHSandRHS;
    const pdb::String &firstTupleSet = rightSources[pipelineSource].firstTupleSet;

    // get the source computation
    auto srcNode = s->logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

    // go grab the source page set
    PDBAbstractPageSetPtr sourcePageSet = rightSourcePageSets[pipelineSource];

    // did we manage to get a source page set? if not the setup failed
    if (sourcePageSet == nullptr) {
      return false;
    }

    /// 14.4.2 Figure out the parameters of the pipeline

    // figure out the join arguments
    auto joinArguments = getJoinArguments(storage);

    // if we could not create them we are out of here
    if (joinArguments == nullptr) {
      return false;
    }

    // mark that this is a join agg side
    joinArguments->isJoinAggSide = true;

    // empty computations parameters
    params = {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<DismissProcessor>()},
              {ComputeInfoType::JOIN_ARGS, joinArguments},
              {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(swapLHSandRHS)},
              {ComputeInfoType::JOIN_AGG_SIDE_ARGS,
               std::make_shared<JoinAggSideArg>(s->rightKeyPage, s->rightJoinSideSenders, s->planPage, false)},
              {ComputeInfoType::SOURCE_SET_INFO, getRightSourceSetArg(catalogClient, pipelineSource)}};

    /// 14.4.3 Build the pipeline

    // build the join pipeline
    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       joinAtomicComp->getRightInput().getSetName(),     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       s->intermediatePageSet,
                                       params,
                                       job->thisNode,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       pipelineIndex);

    // store the join pipeline
    s->joinPipelines->push_back(pipeline);
  }

  /// 15. Setup the aggregation pipeline

  // the join arguments
  auto joinArguments = std::make_shared<JoinArguments>(JoinArgumentsInit{{joinAtomicComp->getRightInput().getSetName(),
                                                                          std::make_shared<JoinArg>(s->rightShuffledPageSet)}});

  // mark that this is the join aggregation algorithm
  joinArguments->isJoinAggAggregation = true;
  joinArguments->isLocalJoinAggAggregation = false;
  joinArguments->emitter = s->emitter;

  // set the page that contains the mapping from aggregation key to tid
  joinArguments->aggKeyPage = s->aggKeyPage;

  // set the left and right mappings
  joinArguments->leftTIDToRecordMapping = &s->leftTIDToRecordMapping;
  joinArguments->rightTIDToRecordMapping = &s->rightTIDToRecordMapping;

  // set the plan page
  joinArguments->planPage = s->planPage;

  /// 15.1 Init the preaggregation queues

  s->pageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for(int i = 0; i < job->numberOfNodes; ++i) { s->pageQueues->emplace_back(std::make_shared<PDBPageQueue>()); }

  // fill uo the vector for each thread
  s->preaggregationPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    /// 15.2. Figure out the parameters of the pipeline

    // initialize the parameters
    params = {{ComputeInfoType::PAGE_PROCESSOR,
               std::make_shared<PreaggregationPageProcessor>(job->numberOfNodes, // we use one since this pipeline is completely local.
                                                             job->numberOfProcessingThreads,
                                                             *s->pageQueues,
                                                             myMgr)},
              {ComputeInfoType::JOIN_ARGS, joinArguments},
              {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(false)},
              {ComputeInfoType::SOURCE_SET_INFO, nullptr}};

    /// 15.3. Build the pipeline

    auto pipeline = plan.buildPipeline(joinTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       s->leftShuffledPageSet,
                                       s->intermediatePageSet,
                                       params,
                                       job->thisNode,
                                       job->numberOfNodes, // we use one since this pipeline is completely local.
                                       job->numberOfProcessingThreads,
                                       pipelineIndex);

    s->preaggregationPipelines->push_back(pipeline);
  }

  /// 8. Create the aggregation pipeline

  // get the sink page set
  auto sinkPageSet =
      storage->createAnonymousPageSet(std::make_pair(sink.pageSetIdentifier.first, sink.pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if (sinkPageSet == nullptr) {
    return false;
  }

  // we are putting the pages from the queues here
  s->preaggPageSet = storage->createFeedingAnonymousPageSet(std::make_pair(preaggIntermediate.pageSetIdentifier.first, (std::string) preaggIntermediate.pageSetIdentifier.second),
                                                            job->numberOfProcessingThreads,
                                                            job->numberOfNodes);

  // did we manage to get a page set where we receive this? if not the setup failed
  if(s->preaggPageSet == nullptr) {
    return false;
  }

  /// 7. Create the self receiver to forward pages that are created on this node and the network senders to forward pages for the other nodes

  int32_t currNode = job->thisNode;
  s->senders = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for(unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if(job->nodes[i]->port == job->nodes[currNode]->port && job->nodes[i]->address == job->nodes[currNode]->address) {

      // make the self receiver
      s->selfReceiver = std::make_shared<pdb::PDBPageSelfReceiver>(s->pageQueues->at(i), s->preaggPageSet, myMgr);
    }
    else {

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(storage->getConMgr(),
                                                           job->nodes[i]->nodeID,
                                                           job->numberOfProcessingThreads,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           s->logger,
                                                           std::make_pair(preaggIntermediate.pageSetIdentifier.first, (std::string) preaggIntermediate.pageSetIdentifier.second),
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
    auto aggPipeline = plan.buildAggregationPipeline(finalTupleSet, myMgr->getWorkerQueue(), s->preaggPageSet, sinkPageSet, workerID);

    // store the aggregation pipeline
    s->aggregationPipelines->push_back(aggPipeline);
  }

  // get the key extractor
  s->keyExtractor = getKeyExtractor(finalTupleSet, plan);

  return true;
}

bool pdb::PDBJoinAggregationComputationStage::run(const pdb::Handle<pdb::ExJob> &job,
                                                  const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                                  const std::shared_ptr<pdb::PDBStorageManager> &storage,
                                                  const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoinAggregationState>(state);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  /**
   * 1. Run join pipelines pipelines
   */

  // stats
  atomic_bool success;
  success = true;

  // create the buzzer
  atomic_int counter;
  counter = 0;
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // run on of the join pipelines
  for (int i = 0; i < s->joinPipelines->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr
        myWork = std::make_shared<pdb::GenericWork>([&counter, &success, i, &s](const PDBBuzzerPtr &callerBuzzer) {

      try {

        // run the pipeline
        (*s->joinPipelines)[i]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  std::cout << "Run the join pipelines\n";

  // create the buzzer
  atomic_int sendCnt;
  sendCnt = 0;

  for (int i = 0; i < s->leftJoinSideSenders->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr
        myWork = std::make_shared<pdb::GenericWork>([&sendCnt, &success, i, s](const PDBBuzzerPtr &callerBuzzer) {

      try {

        // run the pipeline
        (*s->leftJoinSideSenders)[i]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, sendCnt);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  std::cout << "Run the join left senders\n";

  for (int i = 0; i < s->rightJoinSideSenders->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr
        myWork = std::make_shared<pdb::GenericWork>([&sendCnt, &success, i, &s](const PDBBuzzerPtr &callerBuzzer) {

      try {

        // run the pipeline
        (*s->rightJoinSideSenders)[i]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, sendCnt);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  std::cout << "Run the join right pipelines\n";

  // create the buzzer
  atomic_int commCnt;
  commCnt = 0;

  // combine them so I don't write this twice
  std::vector<JoinMapCreatorPtr> joinMapCreators = *s->leftJoinMapCreators;
  joinMapCreators.insert(joinMapCreators.end(), s->rightJoinMapCreators->begin(), s->rightJoinMapCreators->end());

  for (const auto &joinMapCreator : joinMapCreators) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork =
        std::make_shared<pdb::GenericWork>([&commCnt, &success, joinMapCreator, &s](const PDBBuzzerPtr &callerBuzzer) {

          // run the join map creator
          try {
            // run the join map creator
            joinMapCreator->run();
          }
          catch (std::exception &e) {

            // log the error
            s->logger->error(e.what());

            // we failed mark that we have
            success = false;
          }

          // check if the creator succeeded
          if (!joinMapCreator->getSuccess()) {

            // log the error
            s->logger->error(joinMapCreator->getError());

            // we failed mark that we have
            success = false;
          }

          std::cout << "Ended...\n";

          // signal that the run was successful
          callerBuzzer->buzz(PDBAlarm::WorkAllDone, commCnt);
        });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  std::cout << "Run the join map creators\n";

  // run the aggregation pipelines
  atomic_int preaggCnt;
  preaggCnt = 0;

  // stats
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
  std::cout << "Run the aggregation\n";

  // wait to finish the pipelines
  while (counter < s->joinPipelines->size()) {
    tempBuzzer->wait();
  }
  std::cout << "Finished the join pipelines\n";

  // shutdown the senders since the pipelines are done
  for (auto &se : *s->leftJoinSideSenders) {
    se->shutdown();
  }

  for (auto &se : *s->rightJoinSideSenders) {
    se->shutdown();
  }

  std::cout << "Shutdown the senders\n";

  // wait for senders to finish
  while (sendCnt < s->leftJoinSideSenders->size() + s->rightJoinSideSenders->size()) {
    tempBuzzer->wait();
  }

  std::cout << "Run the join senders\n";

  // wait until the senders finish
  while (commCnt < joinMapCreators.size()) {
    tempBuzzer->wait();
  }

  std::cout << "Finished the join map creators\n";

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "JoinSideStage run for " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
            << "[ns]" << '\n';

  s->emitter->printEms();
  s->emitter->end();

  begin = std::chrono::steady_clock::now();

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

    // do we need to extract the keys too
    if(s->keyExtractor != nullptr) {
      // materialize the keys
      sinkPageSet->resetPageSet();
      success = storage->materializeKeys(sinkPageSet,std::make_pair<std::string, std::string>(setsToMaterialize[j].database, setsToMaterialize[j].set), s->keyExtractor) && success;
    }
  }

  return success;

}

void pdb::PDBJoinAggregationComputationStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManager> &storage) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoinAggregationState>(state);

  // clear everything
  s->joinPipelines->clear();
  s->leftJoinSideSenders->clear();
  s->rightJoinSideSenders->clear();
  s->leftJoinMapCreators->clear();
  s->rightJoinMapCreators->clear();
  s->leftJoinSideCommunicatorsOut->clear();
  s->rightJoinSideCommunicatorsOut->clear();
  s->leftJoinSideCommunicatorsIn->clear();
  s->rightJoinSideCommunicatorsIn->clear();

  // remove the intermediate page set
  storage->removePageSet(std::make_pair(preaggIntermediate.pageSetIdentifier.first, (std::string) preaggIntermediate.pageSetIdentifier.second));

  // reset the page sets
  s->leftShuffledPageSet->resetPageSet();
  s->rightShuffledPageSet->resetPageSet();
  s->intermediatePageSet->clearPageSet();
}

pdb::PDBAbstractPageSetPtr pdb::PDBJoinAggregationComputationStage::getRightSourcePageSet(const std::shared_ptr<pdb::PDBStorageManager> &storage,
                                                                                          size_t idx) {
  // grab the source set from the right sources
  auto &sourceSet = this->rightSources[idx].sourceSet;

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet;
  if (sourceSet != nullptr) {

    // get the page set
    sourcePageSet = storage->createPageSetFromPDBSet(sourceSet->database, sourceSet->set, false);
    sourcePageSet->resetPageSet();

  } else {

    // we are reading from an existing page set get it
    sourcePageSet = storage->getPageSet(this->rightSources[idx].pageSet->pageSetIdentifier);
    sourcePageSet->resetPageSet();
  }

  // return the page set
  return sourcePageSet;
}

pdb::SourceSetArgPtr pdb::PDBJoinAggregationComputationStage::getRightSourceSetArg(const std::shared_ptr<pdb::PDBCatalogClient> &catalogClient,
                                                                                   size_t idx) {
  // grab the source set from the sources
  auto &sourceSet = this->rightSources[idx].sourceSet;

  // check if we actually have a set
  if (sourceSet == nullptr) {
    return nullptr;
  }

  // return the argument
  std::string error;
  return std::make_shared<pdb::SourceSetArg>(catalogClient->getSet(sourceSet->database, sourceSet->set, error));
}

pdb::JoinCompBase *pdb::PDBJoinAggregationComputationStage::getJoinComp(const LogicalPlanPtr &logicalPlan) {

  auto &computations = logicalPlan->getComputations();

  // get the join atomic computation computation
  auto joinComp = computations.getProducingAtomicComputation(joinTupleSet);

  // get the real computation
  auto compNode = logicalPlan->getNode(joinComp->getComputationName());
  return ((JoinCompBase *) &logicalPlan->getNode(joinComp->getComputationName()).getComputation());
}
