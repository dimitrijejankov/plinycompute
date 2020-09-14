
#include <physicalAlgorithms/TRAMaterializeOperationState.h>
#include <physicalAlgorithms/TRAMaterializeOperationStage.h>
#include "TRAMaterializeOperation.h"

pdb::TRAMaterializeOperation::TRAMaterializeOperation(const std::string &db, const std::string &set, const std::string &pageSet) : db(db), set(set), pageSet(pageSet) {

  setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::TRAMaterializeOperation::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  return std::make_shared<pdb::TRAMaterializeOperationState>();
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::TRAMaterializeOperation::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
  if(currentStage == 0) {
    currentStage++;
    return std::make_shared<TRAMaterializeOperationStage>(db, set, pageSet);
  }
  return nullptr;
}

int32_t pdb::TRAMaterializeOperation::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::TRAMaterializeOperation::getAlgorithmType() {
  return PDBPhysicalAlgorithmType::IndexOperation;
}

pdb::PDBCatalogSetContainerType pdb::TRAMaterializeOperation::getOutputContainerType() {
  return PDB_CATALOG_SET_VECTOR_CONTAINER;
}
