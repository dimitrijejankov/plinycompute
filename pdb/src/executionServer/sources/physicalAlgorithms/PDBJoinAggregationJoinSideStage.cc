#include <AtomicComputationClasses.h>
#include <PDBPhysicalAlgorithm.h>
#include "PDBJoinAggregationJoinSideStage.h"
#include "PDBJoinAggregationState.h"
#include "ComputePlan.h"
#include "LogicalPlanTransformer.h"
#include "DismissProcessor.h"
#include "PreaggregationPageProcessor.h"
#include "GenericWork.h"
#include "ExJob.h"
//#include "../../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlock.h"

pdb::PDBJoinAggregationJoinSideStage::PDBJoinAggregationJoinSideStage(const pdb::PDBSinkPageSetSpec &sink,
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
      rightSources(right_sources) {}

bool pdb::PDBJoinAggregationJoinSideStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                                 const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                                 const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                 const std::string &error) {


  // cast the state
  auto s = dynamic_pointer_cast<PDBJoinAggregationState>(state);

  // get catalog client
  auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  auto logicalPlan = plan.getPlan();

  // make the transformer
  auto transformer = std::make_shared<LogicalPlanTransformer>(logicalPlan);
  transformer->addTransformation(std::make_shared<DropToKeyExtractionTransformation>(joinTupleSet));

  // apply all the transformations
  logicalPlan = transformer->applyTransformations();

  // get the join comp
  auto joinAtomicComp = dynamic_pointer_cast<ApplyJoin>(logicalPlan->getComputations().getProducingAtomicComputation(joinTupleSet));

  std::cout << "Exec plan" << *logicalPlan << '\n';

  /// 10. Make outgoing connections to other nodes

  // make the object
  UseTemporaryAllocationBlock tmp{1024};
  pdb::Handle<SerConnectToRequest> connectionID = pdb::makeObject<SerConnectToRequest>(job->computationID,
                                                                                       job->jobID,
                                                                                       job->thisNode,
                                                                                       PDBJoinAggregationState::LEFT_JOIN_SIDE_TASK);

  // init the vector for the left sides
  s->leftJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for(int n = 0; n < job->numberOfNodes; n++) {

    // connect to the node
    s->leftJoinSideCommunicatorsOut->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                                job->nodes[n]->backendPort,
                                                                connectionID));
  }

  // init the vector for the right sides
  connectionID->taskID = PDBJoinAggregationState::RIGHT_JOIN_SIDE_TASK;
  s->rightJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for(int n = 0; n < job->numberOfNodes; n++) {

    // connect to the node
    s->rightJoinSideCommunicatorsOut->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                                 job->nodes[n]->backendPort,
                                                                 connectionID));
  }

  /// 11. Get the incoming connections to this node.

  // wait for left side connections
  connectionID->taskID = PDBJoinAggregationState::LEFT_JOIN_SIDE_TASK;
  s->leftJoinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for(int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // wait for the connection
    s->leftJoinSideCommunicatorsIn->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if(s->leftJoinSideCommunicatorsIn->back()->isSocketClosed()) {

      // log the error
      s->logger->error("Socket for the left side is closed");

      return false;
    }
  }

  // wait for the right side connections
  connectionID->taskID = PDBJoinAggregationState::RIGHT_JOIN_SIDE_TASK;
  s->rightJoinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for(int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // wait for the connection
    s->rightJoinSideCommunicatorsIn->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if(s->rightJoinSideCommunicatorsIn->back()->isSocketClosed()) {

      // log the error
      s->logger->error("Socket for the right side is closed");

      return false;
    }
  }

  /// 12. Setup the join side senders

  // get the join computation
  auto joinComp = getJoinComp(logicalPlan);

  // setup the left senders
  s->leftJoinSideSenders = std::make_shared<std::vector<JoinAggSideSenderPtr>>();

  // init the senders
  for(auto &comm : *s->leftJoinSideCommunicatorsIn) {

    // init the right senders
    s->leftJoinSideSenders->emplace_back(joinComp->getJoinAggSender(joinAtomicComp->getProjection(),
                                                                    logicalPlan,
                                                                    myMgr->getPage(),
                                                                    comm));
  }

  // setup the right senders
  s->rightJoinSideSenders = std::make_shared<std::vector<JoinAggSideSenderPtr>>();

  // init the senders
  for(auto &comm : *s->rightJoinSideCommunicatorsIn) {

    // init the right senders
    s->rightJoinSideSenders->emplace_back(joinComp->getJoinAggSender(joinAtomicComp->getRightProjection(),
                                                                    logicalPlan,
                                                                    myMgr->getPage(),
                                                                    comm));
  }

  /// 13. Setup the join map creators
  s->joinMapCreators = std::make_shared<std::vector<JoinMapCreatorPtr>>();

  // init the join side creators
  s->leftShuffledPageSet = storage->createAnonymousPageSet(leftJoinSource.pageSetIdentifier);
  for(auto &comm : *s->leftJoinSideCommunicatorsOut) {

    // make the creators
    s->joinMapCreators->emplace_back(joinComp->getJoinMapCreator(joinAtomicComp->getProjection(),
                                 logicalPlan,
                                     storage->getConfiguration()->numThreads,
                                     job->thisNode,
                                    true,
                                     s->planPage,
                                     s->leftShuffledPageSet,
                                     comm,
                                     myMgr->getPage(),
                                     s->logger));
  }

  // init the join side creators
  s->rightShuffledPageSet = storage->createAnonymousPageSet(rightJoinSource.pageSetIdentifier);
  for(auto &comm : *s->rightJoinSideCommunicatorsOut) {

    // make the creators
    s->joinMapCreators->emplace_back(joinComp->getJoinMapCreator(joinAtomicComp->getProjection(),
                                                                 logicalPlan,
                                                                 storage->getConfiguration()->numThreads,
                                                                 job->thisNode,
                                                                 false,
                                                                 s->planPage,
                                                                 s->rightShuffledPageSet,
                                                                 comm,
                                                                 myMgr->getPage(),
                                                                 s->logger));
  }

  /// 14. the left and right side of the join

  // the join key pipelines
  s->joinPipelines = make_shared<std::vector<PipelinePtr>>();

  // get the sink page set
  s->intermediatePageSet = storage->createAnonymousPageSet(std::make_pair(intermediateSink.pageSetIdentifier.first,
                                                                                    intermediateSink.pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if(s->intermediatePageSet == nullptr) {
    return false;
  }

  /// 14.1 Initialize the left sources and the sink

  // we put them here
  std::vector<PDBAbstractPageSetPtr> leftSourcePageSets;
  leftSourcePageSets.reserve(sources.size());

  // initialize them
  for(int i = 0; i < sources.size(); i++) {
    leftSourcePageSets.emplace_back(getSourcePageSet(storage, i));
  }

  /// 14.2. For each node initialize left pipelines

  // initialize all the pipelines
  std::map<ComputeInfoType, ComputeInfoPtr> params;
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    /// 14.2.1 Figure out what source to use

    // figure out what pipeline
    auto pipelineSource = pipelineIndex % sources.size();

    // grab these thins from the source we need them
    bool swapLHSandRHS = sources[pipelineSource].swapLHSandRHS;
    const pdb::String &firstTupleSet = sources[pipelineSource].firstTupleSet;

    // get the source computation
    auto srcNode = logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

    // go grab the source page set
    PDBAbstractPageSetPtr sourcePageSet = leftSourcePageSets[pipelineSource];

    // did we manage to get a source page set? if not the setup failed
    if(sourcePageSet == nullptr) {
      return false;
    }

    /// 14.2.2 Figure out the parameters of the pipeline

    // figure out the join arguments
    auto joinArguments = getJoinArguments (storage);

    // if we could not create them we are out of here
    if(joinArguments == nullptr) {
      return false;
    }

    // mark that this is a join agg side
    joinArguments->isJoinAggSide = true;

    // empty computations parameters
    params =  {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<DismissProcessor>()},
               {ComputeInfoType::JOIN_ARGS, joinArguments},
               {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(swapLHSandRHS)},
               {ComputeInfoType::JOIN_AGG_SIDE_ARGS, std::make_shared<JoinAggSideArg>(s->leftKeyPage, s->leftJoinSideSenders, s->planPage, true)},
               {ComputeInfoType::SOURCE_SET_INFO, getSourceSetArg(catalogClient, pipelineSource)}};

    /// 14.2.3 Build the pipeline

    // build the join pipeline
    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       joinAtomicComp->getInputName(),     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       s->intermediatePageSet,
                                       params,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       20,
                                       pipelineIndex);

    // store the join pipeline
    s->joinPipelines->push_back(pipeline);
  }

  /// 14.3 Initialize the right sources

  // we put them here
  std::vector<PDBAbstractPageSetPtr> rightSourcePageSets;
  rightSourcePageSets.reserve(sources.size());

  // initialize them
  for(int i = 0; i < rightSources.size(); i++) {
    rightSourcePageSets.emplace_back(getRightSourcePageSet(storage, i));
  }

  /// 14.4. For each node initialize left pipelines

  // initialize all the pipelines
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    /// 14.4.1 Figure out what source to use

    // figure out what pipeline
    auto pipelineSource = pipelineIndex % rightSources.size();

    // grab these thins from the source we need them
    bool swapLHSandRHS = rightSources[pipelineSource].swapLHSandRHS;
    const pdb::String &firstTupleSet = rightSources[pipelineSource].firstTupleSet;

    // get the source computation
    auto srcNode = logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

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
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       20,
                                       pipelineIndex);

    // store the join pipeline
    s->joinPipelines->push_back(pipeline);
  }

  return true;
}

bool pdb::PDBJoinAggregationJoinSideStage::run(const pdb::Handle<pdb::ExJob> &job,
                                               const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                               const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
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
    if(myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // run on of the join pipelines
  counter = 0;
  for (int i = 0; i < s->joinPipelines->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&counter, &success, i, &s](const PDBBuzzerPtr& callerBuzzer) {

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

  // create the buzzer
  atomic_int sendCnt;
  sendCnt = 0;

  for (int i = 0; i < s->leftJoinSideSenders->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendCnt, &success, i, s](const PDBBuzzerPtr& callerBuzzer) {

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

  for (int i = 0; i < s->rightJoinSideSenders->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendCnt, &success, i, &s](const PDBBuzzerPtr& callerBuzzer) {

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

  // create the buzzer
  atomic_int commCnt;
  commCnt = 0;

  for(int i = 0; i < s->joinMapCreators->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&commCnt, &success, i, &s](const PDBBuzzerPtr& callerBuzzer) {

      // run the join map creator
      try {
        // run the join map creator
        (*s->joinMapCreators)[i]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // check if the creator succeeded
      if(!(*s->joinMapCreators)[i]->getSuccess()) {

        // log the error
        s->logger->error((*s->joinMapCreators)[i]->getError());

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

  // wait to finish the pipelines
  while (counter < s->joinPipelines->size()) {
    tempBuzzer->wait();
  }

  // shutdown the senders since the pipelines are done
  for(auto &se : *s->leftJoinSideSenders) {
    se->shutdown();
  }

  for(auto &se : *s->rightJoinSideSenders) {
    se->shutdown();
  }

  // wait for senders to finish
  while (sendCnt < s->leftJoinSideSenders->size() + s->rightJoinSideSenders->size()) {
    tempBuzzer->wait();
  }

  // wait until the senders finish
  while (commCnt < s->joinMapCreators->size()) {
    tempBuzzer->wait();
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Run pipeline for " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << '\n';

  return true;
}

void pdb::PDBJoinAggregationJoinSideStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoinAggregationState>(state);

  // clear everything
  s->joinPipelines->clear();
  s->leftJoinSideSenders->clear();
  s->rightJoinSideSenders->clear();
  s->joinMapCreators->clear();
  s->leftJoinSideCommunicatorsOut->clear();
  s->rightJoinSideCommunicatorsOut->clear();
  s->leftJoinSideCommunicatorsIn->clear();
  s->rightJoinSideCommunicatorsIn->clear();

  // reset the page sets
  s->leftShuffledPageSet->resetPageSet();
  s->leftShuffledPageSet->setAccessOrder(PDBAnonymousPageSetAccessPattern::CONCURRENT);
  s->rightShuffledPageSet->resetPageSet();
  s->rightShuffledPageSet->setAccessOrder(PDBAnonymousPageSetAccessPattern::CONCURRENT);
  s->intermediatePageSet->clearPageSet();
}

pdb::PDBAbstractPageSetPtr pdb::PDBJoinAggregationJoinSideStage::getRightSourcePageSet(const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
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

pdb::SourceSetArgPtr pdb::PDBJoinAggregationJoinSideStage::getRightSourceSetArg(const std::shared_ptr<pdb::PDBCatalogClient> &catalogClient,
                                                                                size_t idx) {
  // grab the source set from the sources
  auto &sourceSet = this->rightSources[idx].sourceSet;

  // check if we actually have a set
  if(sourceSet == nullptr) {
    return nullptr;
  }

  // return the argument
  std::string error;
  return std::make_shared<pdb::SourceSetArg>(catalogClient->getSet(sourceSet->database, sourceSet->set, error));
}

pdb::JoinCompBase* pdb::PDBJoinAggregationJoinSideStage::getJoinComp(const LogicalPlanPtr &logicalPlan) {

  auto &computations = logicalPlan->getComputations();

  // get the join atomic computation computation
  auto joinComp = computations.getProducingAtomicComputation(joinTupleSet);

  // get the real computation
  auto compNode = logicalPlan->getNode(joinComp->getComputationName());
  return ((JoinCompBase*) &logicalPlan->getNode(joinComp->getComputationName()).getComputation());
}