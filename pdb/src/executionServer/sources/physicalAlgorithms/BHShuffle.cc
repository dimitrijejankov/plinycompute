#include "BHShuffle.h"
#include "BHShuffleState.h"
#include "BHShuffleStage.h"

pdb::BHShuffle::BHShuffle(const std::string &db,
                          const std::string &set,
                          const std::vector<int32_t> &indices,
                          const std::string &sink) : indices(indices.size(), indices.size()){
  for(int i = 0; i < indices.size(); ++i) {
    this->indices[i] = indices[i];
  }
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::BHShuffle::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  return std::make_shared<BHShuffleState>();
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::BHShuffle::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
  if(currentStage == 0) {
    currentStage++;
    return std::make_shared<BHShuffleStage>(db, set, inputPageSet, indices);
  }
  return nullptr;
}

int32_t pdb::BHShuffle::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::BHShuffle::getAlgorithmType() {
  return pdb::PDBPhysicalAlgorithmType::Shuffle;
}

pdb::PDBCatalogSetContainerType pdb::BHShuffle::getOutputContainerType() {
  return PDB_CATALOG_SET_VECTOR_CONTAINER;
}