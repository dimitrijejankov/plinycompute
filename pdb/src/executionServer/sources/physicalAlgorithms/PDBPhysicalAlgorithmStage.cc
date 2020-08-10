#include <PDBPhysicalAlgorithmStage.h>
#include <PDBPhysicalAlgorithm.h>
#include <ComputePlan.h>

namespace pdb {

PDBAbstractPageSetPtr PDBPhysicalAlgorithmStage::getSourcePageSet(const std::shared_ptr<pdb::PDBStorageManagerFrontend> &storage, size_t idx) {

  // grab the source set from the sources
  auto &sourceSet = this->sources[idx].sourceSet;

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet;
  if (sourceSet != nullptr) {

    // get the page set
    sourcePageSet = storage->createPageSetFromPDBSet(sourceSet->database, sourceSet->set, false);
    sourcePageSet->resetPageSet();

  } else {

    // we are reading from an existing page set get it
    sourcePageSet = storage->getPageSet(this->sources[idx].pageSet->pageSetIdentifier);
    sourcePageSet->resetPageSet();
  }

  // return the page set
  return sourcePageSet;
}

pdb::SourceSetArgPtr PDBPhysicalAlgorithmStage::getSourceSetArg(const std::shared_ptr<pdb::PDBCatalogClient> &catalogClient, size_t idx) {

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

std::shared_ptr<JoinArguments> PDBPhysicalAlgorithmStage::getJoinArguments(const std::shared_ptr<pdb::PDBStorageManagerFrontend> &storage) {

  // go through each of the additional sources and add them to the join arguments
  auto joinArguments = std::make_shared<JoinArguments>();
  for(int i = 0; i < this->secondarySources.size(); ++i) {

    // grab the source identifier and with it the page set of the additional source
    auto &sourceIdentifier = *(this->secondarySources[i]);
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

PDBKeyExtractorPtr PDBPhysicalAlgorithmStage::getKeyExtractor(const std::string &tupleSet, ComputePlan &plan) {

  // find the target atomic computation
  auto targetAtomicComp = plan.getPlan()->getComputations().getProducingAtomicComputation(tupleSet);

  // find the target real PDBComputation
  auto targetComputationName = targetAtomicComp->getComputationName();

  // grab the aggregation combiner
  Handle<Computation> agg = unsafeCast<Computation>(plan.getPlan()->getNode(targetComputationName).getComputationHandle());

  // get the key extractor
  return agg->getKeyExtractor();
}

}