#include "LocalJoin.h"
#include "LocalJoinState.h"
#include "LocalJoinStage.h"

pdb::LocalJoin::LocalJoin(const std::string &lhsPageSet,
                          const std::vector<int32_t>& lhs_indices,
                          const std::string &rhsDb,
                          const std::string &rhsSet,
                          const std::vector<int32_t>& rhs_indices,
                          const std::string &pageSet) : lhs_indices(lhs_indices.size(), lhs_indices.size()){
  for(int i = 0; i < lhs_indices.size(); ++i) {
    this->lhs_indices[i] = lhs_indices[i];
  }
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::LocalJoin::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  return std::make_shared<LocalJoinState>();
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::LocalJoin::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
  if(currentStage == 0) {
    currentStage++;
    return std::make_shared<LocalJoinStage>(db, set, inputPageSet, lhs_indices);
  }
  return nullptr;
}

int32_t pdb::LocalJoin::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::LocalJoin::getAlgorithmType() {
  return pdb::PDBPhysicalAlgorithmType::LocalJoinType;
}

pdb::PDBCatalogSetContainerType pdb::LocalJoin::getOutputContainerType() {
  return PDB_CATALOG_SET_VECTOR_CONTAINER;
}