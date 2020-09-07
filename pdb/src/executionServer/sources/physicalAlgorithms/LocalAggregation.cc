#include "LocalAggregation.h"
#include "LocalAggregationState.h"
#include "LocalAggregationStage.h"

pdb::LocalAggregation::LocalAggregation(const std::string &db,
                          const std::string &set,
                          const std::vector<int32_t> &indices,
                          const std::string &sink) : indices(indices.size(), indices.size()){
  for(int i = 0; i < indices.size(); ++i) {
    this->indices[i] = indices[i];
  }
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::LocalAggregation::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  return std::make_shared<LocalAggregationState>();
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::LocalAggregation::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
  if(currentStage == 0) {
    currentStage++;
    return std::make_shared<LocalAggregationStage>(db, set, inputPageSet, indices);
  }
  return nullptr;
}

int32_t pdb::LocalAggregation::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::LocalAggregation::getAlgorithmType() {
  return pdb::PDBPhysicalAlgorithmType::JoinAggregation;
}

pdb::PDBCatalogSetContainerType pdb::LocalAggregation::getOutputContainerType() {
  return PDB_CATALOG_SET_VECTOR_CONTAINER;
}