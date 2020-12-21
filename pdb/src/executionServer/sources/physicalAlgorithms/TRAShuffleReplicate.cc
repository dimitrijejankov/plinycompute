#include "TRAShuffleReplicate.h"
#include "TRAShuffleReplicateState.h"
#include "TRAShuffleReplicateStage.h"

pdb::TRAShuffleReplicate::TRAShuffleReplicate(const std::string &inputPageSet, int32_t newIdx, int32_t numRepl,
                                              const std::vector<int32_t> &indices, const std::string &sink) :
                                                       indices(indices.size(), indices.size()),
                                                       inputPageSet(inputPageSet),
                                                       sink(sink),
                                                       newIdx(newIdx),
                                                       numRepl(numRepl) {
  for(int i = 0; i < indices.size(); ++i) {
    this->indices[i] = indices[i];
  }

  // init the sets to materialize
  setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::TRAShuffleReplicate::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  return std::make_shared<TRAShuffleReplicateState>();
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::TRAShuffleReplicate::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
  if(currentStage == 0) {
    currentStage++;
    return std::make_shared<TRAShuffleReplicateStage>(inputPageSet, newIdx, numRepl, sink, indices);
  }
  return nullptr;
}

int32_t pdb::TRAShuffleReplicate::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::TRAShuffleReplicate::getAlgorithmType() {
  return pdb::PDBPhysicalAlgorithmType::Shuffle;
}

pdb::PDBCatalogSetContainerType pdb::TRAShuffleReplicate::getOutputContainerType() {
  return PDB_CATALOG_SET_VECTOR_CONTAINER;
}