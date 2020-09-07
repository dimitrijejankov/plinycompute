#include "TRALocalAggregation.h"
#include "TRALocalAggregationState.h"
#include "TRALocalAggregationStage.h"

pdb::TRALocalAggregation::TRALocalAggregation(const std::string &db,
                                              const std::string &set,
                                              const std::vector<int32_t> &indices,
                                              const std::string &sink) : indices(indices.size(), indices.size()){
  for(int i = 0; i < indices.size(); ++i) {
    this->indices[i] = indices[i];
  }
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::TRALocalAggregation::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  return std::make_shared<TRALocalAggregationState>();
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::TRALocalAggregation::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
  if(currentStage == 0) {
    currentStage++;
    return std::make_shared<TRALocalAggregationStage>(db, set, inputPageSet, indices);
  }
  return nullptr;
}

int32_t pdb::TRALocalAggregation::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::TRALocalAggregation::getAlgorithmType() {
  return pdb::PDBPhysicalAlgorithmType::JoinAggregation;
}

pdb::PDBCatalogSetContainerType pdb::TRALocalAggregation::getOutputContainerType() {
  return PDB_CATALOG_SET_VECTOR_CONTAINER;
}