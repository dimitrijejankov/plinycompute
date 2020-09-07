#include <physicalAlgorithms/BHBroadcastStage.h>
#include <physicalAlgorithms/BHBroadcastState.h>
#include "BHBroadcast.h"

pdb::BHBroadcast::BHBroadcast(const std::string &db, const std::string &set, const std::string &sink) :
    db(db), set(set), sink(sink), inputPageSet("") {

  setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::BHBroadcast::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  return std::make_shared<pdb::BHBroadcastState>();
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::BHBroadcast::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
  if(currentStage == 0) {
    currentStage++;
    return std::make_shared<BHBroadcastStage>(db, set, inputPageSet);
  }
  return nullptr;
}

int32_t pdb::BHBroadcast::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::BHBroadcast::getAlgorithmType() {
  return PDBPhysicalAlgorithmType::Broadcast;
}

pdb::PDBCatalogSetContainerType pdb::BHBroadcast::getOutputContainerType() {
  return PDB_CATALOG_SET_VECTOR_CONTAINER;
}
