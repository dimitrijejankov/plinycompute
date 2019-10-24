#include "PDBJoinAggregationAlgorithm.h"

#include <utility>
#include <AtomicComputationClasses.h>

namespace pdb {

PDBJoinAggregationAlgorithm::PDBJoinAggregationAlgorithm(const std::vector<PDBPrimarySource> &leftSource,
                                                         const std::vector<PDBPrimarySource> &rightSource,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                                         AtomicComputationPtr leftInputTupleSet,
                                                         AtomicComputationPtr rightInputTupleSet,
                                                         AtomicComputationPtr joinTupleSet,
                                                         AtomicComputationPtr aggregationKey,
                                                         const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                                         const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize) : leftInputTupleSet(std::move(leftInputTupleSet)),
                                                                                                                            rightInputTupleSet(std::move(rightInputTupleSet)),
                                                                                                                            joinTupleSet(std::move(joinTupleSet)),
                                                                                                                            aggregationKey(std::move(aggregationKey)) {
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

bool pdb::PDBJoinAggregationAlgorithm::setup(std::shared_ptr<PDBStorageManagerBackend> &storage,
                                             Handle<ExJob> &job,
                                             const std::string &error) {
  return PDBPhysicalAlgorithm::setup(storage, job, error);
}

bool pdb::PDBJoinAggregationAlgorithm::run(std::shared_ptr<PDBStorageManagerBackend> &storage) {
  return PDBPhysicalAlgorithm::run(storage);
}

void pdb::PDBJoinAggregationAlgorithm::cleanup() {
  PDBPhysicalAlgorithm::cleanup();
}

pdb::PDBPhysicalAlgorithmType pdb::PDBJoinAggregationAlgorithm::getAlgorithmType() {
  return JoinAggregation;
}

pdb::PDBCatalogSetContainerType pdb::PDBJoinAggregationAlgorithm::getOutputContainerType() {

  // ends with an aggregation therefore it is a map
  return PDB_CATALOG_SET_MAP_CONTAINER;
}


}
