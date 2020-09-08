#include "TRALocalJoin.h"
#include "TRALocalJoinState.h"
#include "TRALocalJoinStage.h"

pdb::TRALocalJoin::TRALocalJoin(const std::string &lhsPageSet,
                                const std::vector<int32_t>& lhs_indices,
                                const std::string &rhsDb,
                                const std::string &rhsSet,
                                const std::vector<int32_t>& rhs_indices,
                                const std::string &pageSet,
                                const std::string& startPageSet,
                                const std::string& endPageSet) : lhs_indices(lhs_indices.size(), lhs_indices.size()){
  for(int i = 0; i < lhs_indices.size(); ++i) {
    this->lhs_indices[i] = lhs_indices[i];
  }

  // the start and end page set
  this->startPageSet = startPageSet;
  this->endPageSet = endPageSet;

  // init the sets to materialize
  setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::TRALocalJoin::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  return std::make_shared<TRALocalJoinState>();
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::TRALocalJoin::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
  if(currentStage == 0) {
    currentStage++;
    return std::make_shared<TRALocalJoinStage>(db, set, inputPageSet, lhs_indices, startPageSet, endPageSet);
  }
  return nullptr;
}

int32_t pdb::TRALocalJoin::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::TRALocalJoin::getAlgorithmType() {
  return pdb::PDBPhysicalAlgorithmType::LocalJoinType;
}

pdb::PDBCatalogSetContainerType pdb::TRALocalJoin::getOutputContainerType() {
  return PDB_CATALOG_SET_VECTOR_CONTAINER;
}