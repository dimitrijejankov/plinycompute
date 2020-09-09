#include <physicalAlgorithms/TRABroadcastStage.h>
#include <physicalAlgorithms/TRABroadcastState.h>
#include "TRABroadcast.h"

pdb::TRABroadcast::TRABroadcast(const std::string &db, const std::string &set, const std::string &sink) :
    db(db), set(set), sink(sink), inputPageSet("") {

  setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::TRABroadcast::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  return std::make_shared<pdb::TRABroadcastState>();
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::TRABroadcast::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
  if(currentStage == 0) {
    currentStage++;
    return std::make_shared<TRABroadcastStage>(db, set, sink);
  }
  return nullptr;
}

int32_t pdb::TRABroadcast::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::TRABroadcast::getAlgorithmType() {
  return PDBPhysicalAlgorithmType::Broadcast;
}

pdb::PDBCatalogSetContainerType pdb::TRABroadcast::getOutputContainerType() {
  return PDB_CATALOG_SET_VECTOR_CONTAINER;
}
