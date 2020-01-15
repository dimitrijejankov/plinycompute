#include <PDBPhysicalAlgorithm.h>
#include <PDBStorageManagerBackend.h>
#include <AtomicComputationClasses.h>
#include <AtomicComputation.h>
#include <PDBCatalogClient.h>

namespace pdb {


PDBPhysicalAlgorithm::PDBPhysicalAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                                           const AtomicComputationPtr &finalAtomicComputation,
                                           const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                           const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                           const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize) :
                                                                 finalTupleSet(finalAtomicComputation->getOutputName()),
                                                                 sink(sink),
                                                                 setsToMaterialize(setsToMaterialize),
                                                                 sources(primarySource.size(), primarySource.size()) {

  // copy all the primary sources
  for(int i = 0; i < primarySource.size(); ++i) {

    // grab the source
    auto &source = primarySource[i];

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

  // copy all the secondary sources
  this->secondarySources = pdb::makeObject<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>>(secondarySources.size(), 0);
  for(const auto &secondarySource : secondarySources) {
    this->secondarySources->push_back(secondarySource);
  }
}

}