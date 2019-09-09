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

PDBAbstractPageSetPtr PDBPhysicalAlgorithm::getSourcePageSet(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, size_t idx) {

  // grab the source set from the sources
  auto &sourceSet = this->sources[idx].sourceSet;

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet;
  if (sourceSet != nullptr) {

    // get the page set
    std::cout << sourceSet->database << sourceSet->set << "\n";
    sourcePageSet = storage->createPageSetFromPDBSet(sourceSet->database, sourceSet->set);
    sourcePageSet->resetPageSet();

  } else {

    // we are reading from an existing page set get it
    sourcePageSet = storage->getPageSet(this->sources[idx].pageSet->pageSetIdentifier);
    sourcePageSet->resetPageSet();
  }

  // return the page set
  return sourcePageSet;
}

pdb::SourceSetArgPtr PDBPhysicalAlgorithm::getSourceSetArg(std::shared_ptr<pdb::PDBCatalogClient> &catalogClient, size_t idx) {

  // grab the source set from the sources
  auto &sourceSet = this->sources[idx].sourceSet;

  // check if we actually have a set
  if(sourceSet == nullptr) {
    return nullptr;
  }

  // return the argument
  std::string error;
  return std::make_shared<pdb::SourceSetArg>(catalogClient->getSet(sourceSet->database, sourceSet->set, error));
}

std::shared_ptr<JoinArguments> PDBPhysicalAlgorithm::getJoinArguments(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  // go through each of the additional sources and add them to the join arguments
  auto joinArguments = std::make_shared<JoinArguments>();
  for(int i = 0; i < this->secondarySources->size(); ++i) {

    // grab the source identifier and with it the page set of the additional source
    auto &sourceIdentifier = *(*this->secondarySources)[i];
    auto additionalSource = storage->getPageSet(std::make_pair(sourceIdentifier.pageSetIdentifier.first, sourceIdentifier.pageSetIdentifier.second));

    // do we have have a page set for that
    if(additionalSource == nullptr) {
      return nullptr;
    }

    // insert the join argument
    joinArguments->hashTables[sourceIdentifier.pageSetIdentifier.second] = std::make_shared<JoinArg>(additionalSource);
  }

  return joinArguments;
}

}