#include "PDBJoinAggregationAlgorithm.h"

#include <utility>
#include <AtomicComputationClasses.h>
#include <ComputePlan.h>
#include <ExJob.h>
#include <LogicalPlanTransformer.h>

namespace pdb {

PDBJoinAggregationAlgorithm::PDBJoinAggregationAlgorithm(const std::vector<PDBPrimarySource> &leftSource,
                                                         const std::vector<PDBPrimarySource> &rightSource,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &sink,
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

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  logicalPlan = plan.getPlan();

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

  std::cout << *logicalPlan << "\n";

  return true;
}

bool pdb::PDBJoinAggregationAlgorithm::run(std::shared_ptr<PDBStorageManagerBackend> &storage) {
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
