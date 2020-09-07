#include "TRAShuffle.h"
#include "TRAShuffleState.h"
#include "TRAShuffleStage.h"

pdb::TRAShuffle::TRAShuffle(const std::string &db,
                            const std::string &set,
                            const std::vector<int32_t> &indices,
                            const std::string &sink) : indices(indices.size(), indices.size()){
  for(int i = 0; i < indices.size(); ++i) {
    this->indices[i] = indices[i];
  }
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::TRAShuffle::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  return std::make_shared<TRAShuffleState>();
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::TRAShuffle::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
  if(currentStage == 0) {
    currentStage++;
    return std::make_shared<TRAShuffleStage>(db, set, inputPageSet, indices);
  }
  return nullptr;
}

int32_t pdb::TRAShuffle::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::TRAShuffle::getAlgorithmType() {
  return pdb::PDBPhysicalAlgorithmType::Shuffle;
}

pdb::PDBCatalogSetContainerType pdb::TRAShuffle::getOutputContainerType() {
  return PDB_CATALOG_SET_VECTOR_CONTAINER;
}