#include "PDBJoinAggregationAlgorithm.h"

#include <utility>
#include <AtomicComputationClasses.h>
#include <ComputePlan.h>
#include <ExJob.h>
#include <LogicalPlanTransformer.h>
#include "PDBStorageManagerBackend.h"
#include "KeyComputePlan.h"
#include "GenericWork.h"
#include "AggregateCompBase.h"

namespace pdb {

PDBJoinAggregationAlgorithm::PDBJoinAggregationAlgorithm(const std::vector<PDBPrimarySource> &leftSource,
                                                         const std::vector<PDBPrimarySource> &rightSource,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &leftKeySink,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &rightKeySink,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &joinAggKeySink,
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

bool pdb::PDBJoinAggregationAlgorithm::setup(std::shared_ptr<PDBStorageManagerBackend> &storage, Handle<ExJob> &job, const std::string &error) {

  // make a logical plan
  auto logicalPlan = std::make_shared<LogicalPlan>(job->tcap, *job->computations);

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

  /// 0. Figure out the left and righ key sink tuple set tuple set

  // get the left key sink page set
  leftPageSet = storage->createAnonymousPageSet(std::make_pair(lhsKeySink->pageSetIdentifier.first, lhsKeySink->pageSetIdentifier.second));

  // did we manage to get a left key sink page set? if not the setup failed
  if(leftPageSet == nullptr) {
    return false;
  }

  // get the right key sink page set
  rightPageSet = storage->createAnonymousPageSet(std::make_pair(rhsKeySink->pageSetIdentifier.first, rhsKeySink->pageSetIdentifier.second));

  // did we manage to get a right key sink page set? if not the setup failed
  if(rightPageSet == nullptr) {
    return false;
  }

  // get sink for the join aggregation pipeline
  auto joinAggPageSet = storage->createAnonymousPageSet(std::make_pair(joinAggKeySink->pageSetIdentifier.first, joinAggKeySink->pageSetIdentifier.second));

  // did we manage to get the sink for the join aggregation pipeline? if not the setup failed
  if(joinAggPageSet == nullptr) {
    return false;
  }

  /// 1. Initilize the left key pipelines

  // make the compute plan
  auto computePlan = std::make_shared<KeyComputePlan>(logicalPlan);

  // get catalog client
  auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();

  // we put them here
  std::vector<PDBAbstractPageSetPtr> leftSourcePageSets;
  leftSourcePageSets.reserve(sources.size());

  // initialize them
  for(int i = 0; i < sources.size(); i++) {
    leftSourcePageSets.emplace_back(getKeySourcePageSet(storage, i, sources));
  }

  // the join key pipelines
  joinKeyPipelines = make_shared<std::vector<PipelinePtr>>();

  // for each lhs source create a pipeline
  for(uint64_t i = 0; i < sources.size(); i++) {

    // make the parameters for the first set
    std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO,
                                                          getKeySourceSetArg(catalogClient, sources, i)} };

    // go grab the source page set
    PDBAbstractPageSetPtr sourcePageSet = leftSourcePageSets[i];

    // build the pipeline
    auto leftPipeline = computePlan->buildHashPipeline(leftInputTupleSet, sourcePageSet, leftPageSet, params);

    // store the pipeline
    joinKeyPipelines->emplace_back(leftPipeline);
  }

  /// 2. Initialize the right key pipelines

  // we put them here
  std::vector<PDBAbstractPageSetPtr> rightSourcePageSets;
  rightSourcePageSets.reserve(rightSources.size());

  // initialize them
  for(int i = 0; i < rightSources.size(); i++) {
    rightSourcePageSets.emplace_back(getKeySourcePageSet(storage, i, rightSources));
  }

  // for each lhs source create a pipeline
  for(uint64_t i = 0; i < rightSources.size(); i++) {

    // make the parameters for the first set
    std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO,
                                                          getKeySourceSetArg(catalogClient, rightSources, i)} };

    // go grab the source page set
    PDBAbstractPageSetPtr sourcePageSet = rightSourcePageSets[i];

    // build the pipeline
    auto rightPipeline = computePlan->buildHashPipeline(rightInputTupleSet, sourcePageSet, rightPageSet, params);

    // store the pipeline
    joinKeyPipelines->emplace_back(rightPipeline);
  }

  /// 3. Initilize the join-agg key pipeline

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

  // get the left and right key page, basically a map of pdb::Map<Key, uint32_t>
  PDBPageHandle leftKeyPage = myMgr->getPage();
  PDBPageHandle rightKeyPage = myMgr->getPage();

  //
  std::map<ComputeInfoType, ComputeInfoPtr> params;
  params = {{ComputeInfoType::PAGE_PROCESSOR, aggComputation->getAggregationKeyProcessor()},
            {ComputeInfoType::JOIN_ARGS, std::make_shared<JoinArguments>(JoinArgumentsInit{{joinComp->getRightInput().getSetName(), std::make_shared<JoinArg>(rightPageSet)}})},
            {ComputeInfoType::KEY_JOIN_SOURCE_ARGS, std::make_shared<KeyJoinSourceArgs>(std::vector<PDBPageHandle>({leftKeyPage, rightKeyPage}))},
            {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(false)}};

  joinAggPipeline = computePlan->buildJoinAggPipeline(joinComp->getOutputName(),
                                                      sinkComp->getOutputName(),     /* this is the TupleSet the pipeline ends with */
                                                      leftPageSet,
                                                      joinAggPageSet,
                                                      params,
                                                      1,
                                                      1,
                                                      20,
                                                      0);


  return true;
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

bool pdb::PDBJoinAggregationAlgorithm::run(std::shared_ptr<PDBStorageManagerBackend> &storage) {

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
  leftPageSet->resetPageSet();
  rightPageSet->resetPageSet();

  try {

    // run the pipeline
    joinAggPipeline->run();
  }
  catch (std::exception &e) {

    // log the error
    this->logger->error(e.what());

    // we failed mark that we have
    return false;
  }

  return true;
}

void pdb::PDBJoinAggregationAlgorithm::cleanup() {
}

pdb::PDBPhysicalAlgorithmType pdb::PDBJoinAggregationAlgorithm::getAlgorithmType() {
  return JoinAggregation;
}

pdb::PDBCatalogSetContainerType pdb::PDBJoinAggregationAlgorithm::getOutputContainerType() {

  // ends with an aggregation therefore it is a map
  return PDB_CATALOG_SET_MAP_CONTAINER;
}


}
