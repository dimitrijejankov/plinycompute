#include "PDBJoinAggregationAlgorithm.h"

#include <utility>
#include <AtomicComputationClasses.h>
#include <ComputePlan.h>
#include <ExJob.h>
#include <LogicalPlanTransformer.h>
#include <processors/DismissProcessor.h>
#include "PDBStorageManagerBackend.h"
#include "KeyComputePlan.h"
#include "GenericWork.h"
#include "PDBLabeledPageSet.h"
#include "AggregateCompBase.h"

namespace pdb {

PDBJoinAggregationAlgorithm::PDBJoinAggregationAlgorithm(const std::vector<PDBPrimarySource> &leftSource,
                                                         const std::vector<PDBPrimarySource> &rightSource,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &leftKeySink,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &rightKeySink,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &joinAggKeySink,
                                                         const pdb::Handle<PDBSourcePageSetSpec> &leftKeySource,
                                                         const pdb::Handle<PDBSourcePageSetSpec> &rightKeySource,
                                                         const pdb::Handle<PDBSourcePageSetSpec> &planSource,
                                                         const AtomicComputationPtr& leftInputTupleSet,
                                                         const AtomicComputationPtr& rightInputTupleSet,
                                                         const AtomicComputationPtr& joinTupleSet,
                                                         const AtomicComputationPtr& aggregationKey,
                                                         pdb::Handle<PDBSinkPageSetSpec> &hashedLHSKey,
                                                         pdb::Handle<PDBSinkPageSetSpec> &hashedRHSKey,
                                                         pdb::Handle<PDBSinkPageSetSpec> &aggregationTID,
                                                         const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                                         const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize) : hashedLHSKey(hashedLHSKey),
                                                                                                                            hashedRHSKey(hashedRHSKey),
                                                                                                                            aggregationTID(aggregationTID),
                                                                                                                            leftInputTupleSet(leftInputTupleSet->getOutputName()),
                                                                                                                            rightInputTupleSet(rightInputTupleSet->getOutputName()),
                                                                                                                            joinTupleSet(joinTupleSet->getOutputName()) {

  // set the sink
  this->sink = sink;

  // set the key sinks
  this->lhsKeySink = leftKeySink;
  this->rhsKeySink = rightKeySink;
  this->joinAggKeySink = joinAggKeySink;

  // set the key sources
  this->leftKeySource = leftKeySource;
  this->rightKeySource = rightKeySource;
  this->planSource = planSource;

  // set the sets to materialize
  this->setsToMaterialize = setsToMaterialize;

  // set the final tuple set
  finalTupleSet = aggregationKey->getOutputName();

  // ini the source sizes
  sources = pdb::Vector<PDBSourceSpec>(leftSource.size(), leftSource.size());
  rightSources = pdb::Vector<PDBSourceSpec>(rightSource.size(), rightSource.size());

  // copy all the primary sources
  for(int i = 0; i < leftSource.size(); ++i) {

    // grab the source
    auto &source = leftSource[i];

    // check if we are scanning a set if we are fill in sourceSet field
    if(source.startAtomicComputation->getAtomicComputationTypeID() == ScanSetAtomicTypeID) {

      // cast to a scan set
      auto scanSet = (ScanSet*) source.startAtomicComputation.get();

      // get the set info
      sources[i].sourceSet = pdb::makeObject<PDBSetObject>(scanSet->getDBName(), scanSet->getSetName());
    }
    else {
      sources[i].sourceSet = nullptr;
    }

    sources[i].firstTupleSet = source.startAtomicComputation->getOutputName();
    sources[i].pageSet = source.source;
    sources[i].swapLHSandRHS = source.shouldSwapLeftAndRight;
  }

  // copy all the primary sources
  for(int i = 0; i < rightSource.size(); ++i) {

    // grab the source
    auto &source = rightSource[i];

    // check if we are scanning a set if we are fill in sourceSet field
    if(source.startAtomicComputation->getAtomicComputationTypeID() == ScanSetAtomicTypeID) {

      // cast to a scan set
      auto scanSet = (ScanSet*) source.startAtomicComputation.get();

      // get the set info
      rightSources[i].sourceSet = pdb::makeObject<PDBSetObject>(scanSet->getDBName(), scanSet->getSetName());
    }
    else {
      rightSources[i].sourceSet = nullptr;
    }

    rightSources[i].firstTupleSet = source.startAtomicComputation->getOutputName();
    rightSources[i].pageSet = source.source;
    rightSources[i].swapLHSandRHS = source.shouldSwapLeftAndRight;
  }

  // copy all the secondary sources
  this->secondarySources = pdb::makeObject<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>>(secondarySources.size(), 0);
  for(const auto &secondarySource : secondarySources) {
    this->secondarySources->push_back(secondarySource);
  }
}

bool PDBJoinAggregationAlgorithm::setupLead(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                            Handle<pdb::ExJob> &job,
                                            const std::string &error) {
  // make a logical plan
  auto logicalPlan = std::make_shared<LogicalPlan>(job->tcap, *job->computations);

  // init the logger
  logger = std::make_shared<PDBLogger>("PDBJoinAggregationAlgorithm");

  /// 1. Find the aggregation comp

  // figure out the aggregation
  if(!logicalPlan->getComputations().hasConsumer(finalTupleSet)) {
    throw runtime_error("Aggregation key has to have a consumer!");
  }

  // figure out the aggregation computation
  auto consumers = logicalPlan->getComputations().getConsumingAtomicComputations(finalTupleSet);
  if(consumers.size() != 1) {
    throw runtime_error("None or multiple aggregation computations!");
  }
  auto aggComp = consumers.front();

  /// 2. Find the key extraction

  // try to find the key extraction of the key
  auto comps = logicalPlan->getComputations().findByPredicate([&aggComp](AtomicComputationPtr &c){

    if(c->getComputationName() == aggComp->getComputationName() &&
        c->getAtomicComputationTypeID() == ApplyLambdaTypeID) {

      // check if it is a key extraction lambda
      auto it = c->getKeyValuePairs()->find("lambdaType");
      return it != c->getKeyValuePairs()->end() && it->second == "key";
    }

    // check if this is a join
    return false;
  });

  // check if we have exactly one key extraction
  if(comps.size() != 1) {
    throw runtime_error("None or multiple key extractions!");
  }
  auto aggKey = *comps.begin();

  /// 3. Run the transformations

  // make the transformer
  auto transformer = std::make_shared<LogicalPlanTransformer>(logicalPlan);

  // add the transformation
  transformer->addTransformation(std::make_shared<InsertKeyScanSetsTransformation>(leftInputTupleSet));
  transformer->addTransformation(std::make_shared<InsertKeyScanSetsTransformation>(rightInputTupleSet));
  transformer->addTransformation(std::make_shared<JoinKeySideTransformation>(leftInputTupleSet));
  transformer->addTransformation(std::make_shared<JoinKeySideTransformation>(rightInputTupleSet));
  transformer->addTransformation(std::make_shared<JoinKeyTransformation>(joinTupleSet));
  transformer->addTransformation(std::make_shared<DropDependents>(aggComp->getOutputName()));
  transformer->addTransformation(std::make_shared<AggKeyTransformation>(aggKey->getOutputName()));
  transformer->addTransformation(std::make_shared<AddJoinTID>(joinTupleSet));

  // apply all the transformations
  logicalPlan = transformer->applyTransformations();
  std::cout << "Key only plan : \n" << *logicalPlan << '\n';

  // make the compute plan
  auto computePlan = std::make_shared<KeyComputePlan>(logicalPlan);

  // get catalog client
  auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();

  /// 0. Figure out the left and right key sink tuple set tuple set

  // get the left key sink page set
  auto leftPageSet = storage->createAnonymousPageSet(std::make_pair(lhsKeySink->pageSetIdentifier.first, lhsKeySink->pageSetIdentifier.second));

  // did we manage to get a left key sink page set? if not the setup failed
  if(leftPageSet == nullptr) {
    return false;
  }

  // create a labeled page set from it
  labeledLeftPageSet = std::make_shared<PDBLabeledPageSet>(leftPageSet);

  // get the right key sink page set
  auto rightPageSet = storage->createAnonymousPageSet(std::make_pair(rhsKeySink->pageSetIdentifier.first, rhsKeySink->pageSetIdentifier.second));

  // did we manage to get a right key sink page set? if not the setup failed
  if(rightPageSet == nullptr) {
    return false;
  }

  // create a labeled page set from it
  labeledRightPageSet = std::make_shared<PDBLabeledPageSet>(rightPageSet);

  // get sink for the join aggregation pipeline
  joinAggPageSet = storage->createAnonymousPageSet(std::make_pair(joinAggKeySink->pageSetIdentifier.first, joinAggKeySink->pageSetIdentifier.second));

  // did we manage to get the sink for the join aggregation pipeline? if not the setup failed
  if(joinAggPageSet == nullptr) {
    return false;
  }

  /// 1. Initialize the left key pipelines

  // the join key pipelines
  int32_t currNode = job->thisNode;
  joinKeyPipelines = make_shared<std::vector<PipelinePtr>>();

  // for each node we create a pipeline
  for(int n = 0; n < job->numberOfNodes; n++) {

    // for each lhs source create a pipeline
    for(uint64_t i = 0; i < sources.size(); i++) {

      // make the parameters for the first set
      std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO,
                                                            getKeySourceSetArg(catalogClient, sources, i)} };

      // grab the left page set
      auto lps = labeledLeftPageSet->getLabeledView(n);

      // go grab the source page set
      bool isThisNode = job->nodes[currNode]->address == job->nodes[n]->address && job->nodes[currNode]->port == job->nodes[n]->port;
      PDBAbstractPageSetPtr sourcePageSet = isThisNode ? getKeySourcePageSet(storage, i, sources) :
                                                         getFetchingPageSet(storage, i,sources, job->nodes[n]->address, job->nodes[n]->port);

      // build the pipeline
      auto leftPipeline = computePlan->buildHashPipeline(leftInputTupleSet, sourcePageSet, lps, params);

      // store the pipeline
      joinKeyPipelines->emplace_back(leftPipeline);
    }
  }

  /// 2. Initialize the right key pipelines

  // for each node we create a pipeline
  for(int n = 0; n < job->numberOfNodes; n++) {

    // for each lhs source create a pipeline
    for (uint64_t i = 0; i < rightSources.size(); i++) {

      // make the parameters for the first set
      std::map<ComputeInfoType, ComputeInfoPtr> params = {{ComputeInfoType::SOURCE_SET_INFO,
                                                           getKeySourceSetArg(catalogClient, rightSources, i)}};

      // grab the left page set
      auto rps = labeledRightPageSet->getLabeledView(n);

      // go grab the source page set
      bool isThisNode = job->nodes[currNode]->address == job->nodes[n]->address && job->nodes[currNode]->port == job->nodes[n]->port;
      PDBAbstractPageSetPtr sourcePageSet = isThisNode ? getKeySourcePageSet(storage, i, rightSources) :
                                                         getFetchingPageSet(storage, i,rightSources, job->nodes[n]->address, job->nodes[n]->port);

      // build the pipeline
      auto rightPipeline = computePlan->buildHashPipeline(rightInputTupleSet, sourcePageSet, rps, params);

      // store the pipeline
      joinKeyPipelines->emplace_back(rightPipeline);
    }
  }

  /// 3. Initialize the join-agg key pipeline

  auto &computations = logicalPlan->getComputations();

  // try to find the sink
  auto sinkList = computations.findByPredicate([&computations](AtomicComputationPtr &c){

    // check if this is a sink
    return computations.getConsumingAtomicComputations(c->getOutputName()).empty();
  });

  // make sure we have only one join
  if(sinkList.size() != 1){
    throw runtime_error("Could not find an aggregation!");
  }

  // get the join computation
  auto &sinkComp = *sinkList.begin();

  // try to find the join
  auto joinList = computations.findByPredicate([](AtomicComputationPtr &c){

    // check if this is a join
    return c->getAtomicComputationTypeID() == ApplyJoinTypeID;
  });

  // make sure we have only one join
  if(joinList.size() != 1){
    throw runtime_error("Could not find a join!");
  }

  auto &joinComp = *joinList.begin();

  // the key aggregation processor
  auto aggComputation = ((AggregateCompBase*)(&computePlan->getPlan()->getNode(sinkComp->getComputationName()).getComputation()));

  // get the buffer manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // get the left and right key page, basically a map of pdb::Map<Key, TID>
  leftKeyPage = myMgr->getPage();
  rightKeyPage = myMgr->getPage();

  // this page will contain the plan
  planPage = myMgr->getPage();

  // the parameters
  std::map<ComputeInfoType, ComputeInfoPtr> params;
  params = {{ComputeInfoType::PAGE_PROCESSOR, aggComputation->getAggregationKeyProcessor()},
            {ComputeInfoType::JOIN_ARGS, std::make_shared<JoinArguments>(JoinArgumentsInit{{joinComp->getRightInput().getSetName(), std::make_shared<JoinArg>(labeledRightPageSet)}})},
            {ComputeInfoType::KEY_JOIN_SOURCE_ARGS, std::make_shared<KeyJoinSourceArgs>(std::vector<PDBPageHandle>({ leftKeyPage, rightKeyPage }))},
            {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(false)}};

  joinKeyAggPipeline = computePlan->buildJoinAggPipeline(joinComp->getOutputName(),
                                                         sinkComp->getOutputName(),     /* this is the TupleSet the pipeline ends with */
                                                      labeledLeftPageSet,
                                                         joinAggPageSet,
                                                         params,
                                                         1,
                                                         1,
                                                         20,
                                                         0);

  /// 4. Init the sending queues

  // this is to transfer left key TIDs
  leftKeyPageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();

  // this is to transfer right key TIDs
  rightKeyPageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();

  // this is to transfer the plan
  planPageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();

  /// 5. Init the senders

  // setup the senders for the left key page
  if(!setupSenders(job, leftKeySource, storage, leftKeyPageQueues, leftKeySenders, nullptr)) {

    // log the error
    logger->error("Failed to setup the senders for the left keys");

    // return false
    return false;
  }

  // setup the senders for the right key page
  if(!setupSenders(job, rightKeySource, storage, rightKeyPageQueues, rightKeySenders, nullptr)) {

    // log the error
    logger->error("Failed to setup the senders for the right keys");

    // return false
    return false;
  }

  // setup the senders for the plan
  if(!setupSenders(job, planSource, storage, planPageQueues, planSenders, nullptr)) {

    // log the error
    logger->error("Failed to setup the senders for the plan");

    // return false
    return false;
  }

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  logicalPlan = plan.getPlan();

  // make the transformer
  transformer = std::make_shared<LogicalPlanTransformer>(logicalPlan);
  transformer->addTransformation(std::make_shared<DropToKeyExtractionTransformation>(joinTupleSet));

  // apply all the transformations
  logicalPlan = transformer->applyTransformations();

  // get the join comp
  auto joinAtomicComp = dynamic_pointer_cast<ApplyJoin>(logicalPlan->getComputations().getProducingAtomicComputation(joinTupleSet));

  /// 6. the left and right side of the join

  // the join key pipelines
  joinPipelines = make_shared<std::vector<PipelinePtr>>();

  /// 6.1 Initialize the left sources and the sink

  // we put them here
  std::vector<PDBAbstractPageSetPtr> leftSourcePageSets;
  leftSourcePageSets.reserve(sources.size());

  // initialize them
  for(int i = 0; i < sources.size(); i++) {
    leftSourcePageSets.emplace_back(getSourcePageSet(storage, i));
  }

  // get the sink page set
  auto sinkPageSet = storage->createAnonymousPageSet(std::make_pair(sink->pageSetIdentifier.first, sink->pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if(sinkPageSet == nullptr) {
    return false;
  }

  /// 6.2. For each node initialize left pipelines

  // initialize all the pipelines
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    /// 6.2.1 Figure out what source to use

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

    /// 6.2.2 Figure out the parameters of the pipeline

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
               {ComputeInfoType::JOIN_AGG_SIDE_ARGS, std::make_shared<JoinAggSideArg>(leftKeyPage, planPage, true)},
               {ComputeInfoType::SOURCE_SET_INFO, getSourceSetArg(catalogClient, pipelineSource)}};

    /// 6.2.3 Build the pipeline

    std::cout << "Modified only plan : \n" << *logicalPlan << '\n';
    // build the join pipeline
    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       joinAtomicComp->getInputName(),     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       sinkPageSet,
                                       params,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       20,
                                       pipelineIndex);

    // store the join pipeline
    joinPipelines->push_back(pipeline);
  }

  // make the object
  UseTemporaryAllocationBlock tmp{1024};
  pdb::Handle<SerConnectToRequest> connectionID = pdb::makeObject<SerConnectToRequest>(job->computationID,
                                                                                       job->jobID,
                                                                                       job->thisNode,
                                                                                       LEFT_JOIN_SIDE_TASK);

  // init the vector for the left sides
  leftJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for(int n = 0; n < job->numberOfNodes; n++) {

    // connect to the node
    leftJoinSideCommunicatorsOut->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                             job->nodes[n]->backendPort,
                                                             connectionID));
  }

  // init the vector for the right sides
  connectionID->taskID = RIGHT_JOIN_SIDE_TASK;
  rightJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for(int n = 0; n < job->numberOfNodes; n++) {

    // connect to the node
    rightJoinSideCommunicatorsOut->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                              job->nodes[n]->backendPort,
                                                              connectionID));
  }

  // wait for left side connections
  connectionID->taskID = LEFT_JOIN_SIDE_TASK;
  leftJoinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for(int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // wait for the connection
    leftJoinSideCommunicatorsIn->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if(leftJoinSideCommunicatorsIn->back()->isSocketClosed()) {

      // log the error
      logger->error("Socket for the left side is closed");

      return false;
    }
  }

  // wait for the right side connections
  connectionID->taskID = RIGHT_JOIN_SIDE_TASK;
  rightJoinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for(int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // wait for the connection
    rightJoinSideCommunicatorsIn->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if(rightJoinSideCommunicatorsIn->back()->isSocketClosed()) {

      // log the error
      logger->error("Socket for the right side is closed");

      return false;
    }
  }

  // we succeeded
  return true;
}

bool PDBJoinAggregationAlgorithm::setupFollower(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                Handle<pdb::ExJob> &job,
                                                const std::string &error) {
  // init the logger
  logger = std::make_shared<PDBLogger>("PDBJoinAggregationAlgorithm");

  /// 1. Wait to get the plan

  // create to receive the left key page
  leftKeyToNodePageSet = storage->createFeedingAnonymousPageSet(std::make_pair(leftKeySource->pageSetIdentifier.first,
                                                                               leftKeySource->pageSetIdentifier.second),
                                                                1,
                                                                job->numberOfNodes);

  // create to receive the right key page
  rightKeyToNodePageSet = storage->createFeedingAnonymousPageSet(std::make_pair(rightKeySource->pageSetIdentifier.first,
                                                                               rightKeySource->pageSetIdentifier.second),
                                                                 1,
                                                                 job->numberOfNodes);

  // create to receive the plan
  planPageSet = storage->createFeedingAnonymousPageSet(std::make_pair(planSource->pageSetIdentifier.first,
                                                                      planSource->pageSetIdentifier.second),
                                                       1,
                                                       job->numberOfNodes);

  /// 2.

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  logicalPlan = plan.getPlan();

  // make the transformer
  auto transformer = std::make_shared<LogicalPlanTransformer>(logicalPlan);
  transformer->addTransformation(std::make_shared<DropToKeyExtractionTransformation>(joinTupleSet));

  // apply all the transformations
  logicalPlan = transformer->applyTransformations();

  // get the join comp
  auto joinAtomicComp = dynamic_pointer_cast<ApplyJoin>(logicalPlan->getComputations().getProducingAtomicComputation(joinTupleSet));

  /// 3.

  // get catalog client
  auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // the join key pipelines
  joinPipelines = make_shared<std::vector<PipelinePtr>>();

  // get the left and right key page, basically a map of pdb::Map<Key, TID>
  leftKeyPage = myMgr->getPage();
  rightKeyPage = myMgr->getPage();

  // this page will contain the plan
  planPage = myMgr->getPage();

  /// 3.1 Initialize the left sources and the sink

  // we put them here
  std::vector<PDBAbstractPageSetPtr> leftSourcePageSets;
  leftSourcePageSets.reserve(sources.size());

  // initialize them
  for(int i = 0; i < sources.size(); i++) {
    leftSourcePageSets.emplace_back(getSourcePageSet(storage, i));
  }

  // get the sink page set
  auto sinkPageSet = storage->createAnonymousPageSet(std::make_pair(sink->pageSetIdentifier.first, sink->pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if(sinkPageSet == nullptr) {
    return false;
  }

  // initialize all the pipelines
  std::map<ComputeInfoType, ComputeInfoPtr> params;
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    /// 3.2.1 Figure out what source to use

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

    /// 6.2.2 Figure out the parameters of the pipeline

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
               {ComputeInfoType::JOIN_AGG_SIDE_ARGS, std::make_shared<JoinAggSideArg>(leftKeyPage, planPage, true)},
               {ComputeInfoType::SOURCE_SET_INFO, getSourceSetArg(catalogClient, pipelineSource)}};

    /// 6.2.3 Build the pipeline

    std::cout << "Modified only plan : \n" << *logicalPlan << '\n';
    // build the join pipeline
    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       joinAtomicComp->getInputName(),     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       sinkPageSet,
                                       params,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       20,
                                       pipelineIndex);

    // store the join pipeline
    joinPipelines->push_back(pipeline);
  }


  /// Init the connections to other nodes for the join sides

  // make the object
  UseTemporaryAllocationBlock tmp{1024};
  pdb::Handle<SerConnectToRequest> connectionID = pdb::makeObject<SerConnectToRequest>(job->computationID,
                                                                                       job->jobID,
                                                                                       job->thisNode,
                                                                                       LEFT_JOIN_SIDE_TASK);

  // init the vector for the left sides
  leftJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for(int n = 0; n < job->numberOfNodes; n++) {

    // connect to the node
    leftJoinSideCommunicatorsOut->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                             job->nodes[n]->backendPort,
                                                             connectionID));
  }

  // init the vector for the right sides
  connectionID->taskID = RIGHT_JOIN_SIDE_TASK;
  rightJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for(int n = 0; n < job->numberOfNodes; n++) {

    // connect to the node
    rightJoinSideCommunicatorsOut->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                              job->nodes[n]->backendPort,
                                                              connectionID));
  }

  // wait for the connections for the left side
  connectionID->taskID = LEFT_JOIN_SIDE_TASK;
  leftJoinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for(int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // wait for the connection
    leftJoinSideCommunicatorsIn->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if(leftJoinSideCommunicatorsIn->back()->isSocketClosed()) {

      // log the error
      logger->error("Socket for the left side is closed");

      return false;
    }
  }

  // wait for the connections for the right side
  connectionID->taskID = RIGHT_JOIN_SIDE_TASK;
  rightJoinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for(int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // wait for the connection
    rightJoinSideCommunicatorsIn->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if(rightJoinSideCommunicatorsIn->back()->isSocketClosed()) {

      // log the error
      logger->error("Socket for the right side is closed");

      return false;
    }
  }

  return true;
}

bool pdb::PDBJoinAggregationAlgorithm::setup(std::shared_ptr<PDBStorageManagerBackend> &storage, Handle<ExJob> &job, const std::string &error) {

  // check that we have at least one worker per primary source
  if(job->numberOfProcessingThreads < sources.size() ||
     job->numberOfProcessingThreads < rightSources.size()) {
    return false;
  }

  // check if we are the lead node if we are set it up for such
  if(job->isLeadNode) {
    return setupLead(storage, job, error);
  }
  // setup the follower
  else {
    return setupFollower(storage, job, error);
  }
}

PDBAbstractPageSetPtr PDBJoinAggregationAlgorithm::getKeySourcePageSet(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                                       size_t idx,
                                                                       pdb::Vector<PDBSourceSpec> &srcs) {

  // grab the source set from the sources
  auto &sourceSet = srcs[idx].sourceSet;

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet;
  if (sourceSet != nullptr) {

    // get the page set
    sourcePageSet = storage->createPageSetFromPDBSet(sourceSet->database, sourceSet->set, true);
    sourcePageSet->resetPageSet();

  } else {

    // we are reading from an existing page set get it
    sourcePageSet = storage->getPageSet(PDBAbstractPageSet::toKeyPageSetIdentifier(srcs[idx].pageSet->pageSetIdentifier));
    sourcePageSet->resetPageSet();
  }

  // return the page set
  return sourcePageSet;
}

PDBAbstractPageSetPtr PDBJoinAggregationAlgorithm::getFetchingPageSet(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                                      size_t idx,
                                                                      pdb::Vector<PDBSourceSpec> &srcs,
                                                                      const std::string &ip,
                                                                      int32_t port) {

  // grab the source set from the sources
  auto &sourceSet = srcs[idx].sourceSet;

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet;
  if (sourceSet != nullptr) {

    // get the page set
    sourcePageSet = storage->fetchPDBSet(sourceSet->database, sourceSet->set, true, ip, port);
    sourcePageSet->resetPageSet();

  } else {

    // we are reading from an existing page set get it
    sourcePageSet = storage->fetchPageSet(*srcs[idx].pageSet, true, ip, port);
    sourcePageSet->resetPageSet();
  }

  // return the page set
  return sourcePageSet;
}

pdb::SourceSetArgPtr PDBJoinAggregationAlgorithm::getKeySourceSetArg(std::shared_ptr<pdb::PDBCatalogClient> &catalogClient,
                                                                     pdb::Vector<PDBSourceSpec> &sources,
                                                                     size_t idx) {

  // grab the source set from the sources
  auto &sourceSet = sources[idx].sourceSet;

  // check if we actually have a set
  if(sourceSet == nullptr) {
    return nullptr;
  }

  // grab the set
  std::string error;
  auto set = catalogClient->getSet(sourceSet->database, sourceSet->set, error);
  if(set == nullptr || !set->isStoringKeys) {
    return nullptr;
  }

  // update the set so it is keyed
  set->name = PDBCatalog::toKeySetName(sourceSet->set);
  set->containerType = PDB_CATALOG_SET_VECTOR_CONTAINER;

  // return the argument
  return std::make_shared<pdb::SourceSetArg>(set);
}

bool pdb::PDBJoinAggregationAlgorithm::run(std::shared_ptr<PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job) {

  // check if this is the lead node if so run the lead
  if (job->isLeadNode) {
    return runLead(storage, job);
  }
  else {
    // run the follower
    return runFollower(storage, job);
  }
}

bool pdb::PDBJoinAggregationAlgorithm::runFollower(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job){

  /**
   * 1. Wait to left and right join map
   */
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  /// TODO this needs to be rewritten using the new methods for direct communication
  auto tmp = leftKeyToNodePageSet->getNextPage(0);
  memcpy(leftKeyPage->getBytes(), tmp->getBytes(), tmp->getSize());

  /// TODO this needs to be rewritten using the new methods for direct communication
  tmp = rightKeyToNodePageSet->getNextPage(0);
  memcpy(rightKeyPage->getBytes(), tmp->getBytes(), tmp->getSize());

  /**
   * 2. Wait to receive the plan
   */
  /// TODO this needs to be rewritten using the new methods for direct communication
  tmp = planPageSet->getNextPage(0);
  memcpy(planPage->getBytes(), tmp->getBytes(), tmp->getSize());

  /**
   * 3. Run join pipelines pipelines
   */

  // stats
  std::string error;
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
  for (int i = 0; i < joinPipelines->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&counter, &success, i, this](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        (*joinPipelines)[i]->run();
      }
      catch (std::exception &e) {

        // log the error
        this->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // wait to finish the pipelines
  while (counter < joinPipelines->size()) {
    tempBuzzer->wait();
  }

  return true;
}

bool PDBJoinAggregationAlgorithm::runLead(std::shared_ptr<PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job) {

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  /**
   * 1. Process the left and right key side
   */

  std::string error;
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

  // here we get a worker per pipeline and run them all.
  for (int i = 0; i < joinKeyPipelines->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&counter, &success, i, this](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        (*joinKeyPipelines)[i]->run();
      }
      catch (std::exception &e) {

        // log the error
        this->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // wait until all the preaggregationPipelines have completed
  while (counter < joinKeyPipelines->size()) {
    tempBuzzer->wait();
  }

  // reset the left and the right page set
  labeledLeftPageSet->resetPageSet();
  labeledRightPageSet->resetPageSet();

  /**
   * 2. Run the join aggregation pipeline
   */

  try {

    // run the pipeline
    joinKeyAggPipeline->run();
  }
  catch (std::exception &e) {

    // log the error
    this->logger->error(e.what());

    // we failed mark that we have
    return false;
  }

  /**
   * 3. Run the planner
   */

  // make the planner
  JoinAggPlanner planner(joinAggPageSet, job->numberOfNodes,planPage);

  // get a page to store the planning result onto
  auto bufferManager = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // do the planning
  planner.doPlanning();

  /**
   * 4. Broadcast the plan (left key page, right key page)
   */

  /// 4.1 Start the senders

  counter = 0;
  PDBBuzzerPtr broadcastKeyPagesBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if(myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });


  /// 4.2 broadcast the left key page

  for(const auto& q : *leftKeyPageQueues) {
    q->enqueue(leftKeyPage);
    q->enqueue(nullptr);
  }

  /// 4.3 broadcast the right key page

  for(const auto& q : *rightKeyPageQueues) {
    q->enqueue(rightKeyPage);
    q->enqueue(nullptr);
  }

  /// 4.4 run the senders for the left side

  for(const auto &sender : *leftKeySenders) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sender, &success, &counter, this](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        sender->run();
      }
      catch (std::exception &e) {

        // log the error
        this->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  /// 4.5 run the senders for the right side

  for(const auto &sender : *rightKeySenders) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sender, &success, &counter, this](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        sender->run();
      }
      catch (std::exception &e) {

        // log the error
        this->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  /**
   * 5. Broadcast to each node the plan except for this one
   */

  /// 5.1 Fill up the plan queues

  for(const auto& q : *planPageQueues) {
    q->enqueue(planPage);
    q->enqueue(nullptr);
  }

  /// 5.2 Run the senders

  for(const auto &sender : *planSenders) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sender, &success, &counter, this](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        sender->run();
      }
      catch (std::exception &e) {

        // log the error
        this->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  /**
   * 6. Wait for everything to finish
   */

  // wait for all the left senders to finish
  // for all the right senders to finish
  // the plan senders
  while (counter < leftKeySenders->size() + rightKeySenders->size() + planSenders->size()) {
    tempBuzzer->wait();
  }

  // run on of the join pipelines
  counter = 0;
  for (int i = 0; i < joinPipelines->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&counter, &success, i, this](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        (*joinPipelines)[i]->run();
      }
      catch (std::exception &e) {

        // log the error
        this->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // wait to finish the pipelines
  while (counter < joinPipelines->size()) {
    tempBuzzer->wait();
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Run pipeline for " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << '\n';

  return true;
}

bool pdb::PDBJoinAggregationAlgorithm::setupSenders(Handle<pdb::ExJob> &job,
                                                    pdb::Handle<PDBSourcePageSetSpec> &recvPageSet,
                                                    std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                    std::shared_ptr<std::vector<PDBPageQueuePtr>> &pageQueues,
                                                    std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> &senders,
                                                    PDBPageSelfReceiverPtr *selfReceiver) {

  // get the buffer manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // go through the nodes and create the page sets
  int32_t currNode = job->thisNode;
  senders = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for(unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if(job->nodes[i]->port == job->nodes[currNode]->port &&
       job->nodes[i]->address == job->nodes[currNode]->address &&
       selfReceiver != nullptr) {

      // create a new queue
      pageQueues->push_back(std::make_shared<PDBPageQueue>());

      // get the receive page set
      auto pageSet = storage->createFeedingAnonymousPageSet(std::make_pair(recvPageSet->pageSetIdentifier.first,
                                                                           recvPageSet->pageSetIdentifier.second),
                                                                           job->numberOfProcessingThreads,
                                                                           job->numberOfNodes);

      // make sure we can use them all at the same time
      pageSet->setUsagePolicy(PDBFeedingPageSetUsagePolicy::KEEP_AFTER_USED);

      // did we manage to get a page set where we receive this? if not the setup failed
      if(pageSet == nullptr) {
        return false;
      }

      // make the self receiver
      *selfReceiver = std::make_shared<pdb::PDBPageSelfReceiver>(pageQueues->back(), pageSet, myMgr);
    }
    else {

      // create a new queue
      pageQueues->push_back(std::make_shared<PDBPageQueue>());

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(job->nodes[i]->address,
                                                           job->nodes[i]->port,
                                                           job->numberOfProcessingThreads,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           logger,
                                                           std::make_pair(recvPageSet->pageSetIdentifier.first, recvPageSet->pageSetIdentifier.second),
                                                           pageQueues->back());

      // setup the sender, if we fail return false
      if(!sender->setup()) {
        return false;
      }

      // make the sender
      senders->emplace_back(sender);
    }
  }

  return true;
}

void pdb::PDBJoinAggregationAlgorithm::cleanup() {

  // invalidate everything
  leftJoinSideCommunicatorsOut = nullptr;
  labeledLeftPageSet = nullptr;
  labeledRightPageSet = nullptr;
  joinAggPageSet = nullptr;
  leftKeyToNodePageSet = nullptr;
  rightKeyToNodePageSet = nullptr;
  planPageSet = nullptr;
  leftKeyPage = nullptr;
  rightKeyPage = nullptr;
  logicalPlan = nullptr;
  leftKeyPageQueues = nullptr;
  rightKeyPageQueues = nullptr;
  planPageQueues = nullptr;
  leftKeySenders = nullptr;
  rightKeySenders = nullptr;
  planSenders = nullptr;
  joinKeyPipelines = nullptr;
  joinKeyAggPipeline = nullptr;
}

pdb::PDBPhysicalAlgorithmType pdb::PDBJoinAggregationAlgorithm::getAlgorithmType() {
  return JoinAggregation;
}

pdb::PDBCatalogSetContainerType pdb::PDBJoinAggregationAlgorithm::getOutputContainerType() {

  // ends with an aggregation therefore it is a map
  return PDB_CATALOG_SET_MAP_CONTAINER;
}

const int32_t pdb::PDBJoinAggregationAlgorithm::LEFT_JOIN_SIDE_TASK = 0;
const int32_t pdb::PDBJoinAggregationAlgorithm::RIGHT_JOIN_SIDE_TASK = 1;

}
