
#include <physicalAlgorithms/TRAIndexOperationState.h>
#include <physicalAlgorithms/TRAIndexOperationStage.h>
#include "TRAIndexOperation.h"

pdb::TRAIndexOperation::TRAIndexOperation(const std::string &db, const std::string &set) : db(db), set(set) {

  setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::TRAIndexOperation::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  return std::make_shared<pdb::TRAIndexOperationState>();
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::TRAIndexOperation::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
  if(currentStage == 0) {
    currentStage++;
    return std::make_shared<TRAIndexOperationStage>(db, set);
  }
  return nullptr;
}

int32_t pdb::TRAIndexOperation::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::TRAIndexOperation::getAlgorithmType() {
  return PDBPhysicalAlgorithmType::IndexOperation;
}

pdb::PDBCatalogSetContainerType pdb::TRAIndexOperation::getOutputContainerType() {
  return PDB_CATALOG_SET_VECTOR_CONTAINER;
}
