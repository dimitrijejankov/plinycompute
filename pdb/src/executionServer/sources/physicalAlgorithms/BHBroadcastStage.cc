#include "BHBroadcastStage.h"




pdb::BHBroadcastStage::BHBroadcastStage(const std::string &db, const std::string &set, const std::string &sink) :
  PDBPhysicalAlgorithmStage(*(_sink),
                            *(_sources),
                            *(_finalTupleSet),
                            *(_secondarySources),
                            *(_setsToMaterialize)), db(db), set(set), sink(sink) {}

bool pdb::BHBroadcastStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                  const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                  const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                  const std::string &error) {
  std::cout << "setup\n";
  return true;
}

bool pdb::BHBroadcastStage::run(const pdb::Handle<pdb::ExJob> &job,
                                const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                const std::string &error) {
  std::cout << "run\n";
  return true;
}

void pdb::BHBroadcastStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                    const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  std::cout << "cleanup\n";
}

const pdb::PDBSinkPageSetSpec *pdb::BHBroadcastStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::BHBroadcastStage::_sources = nullptr;
const pdb::String *pdb::BHBroadcastStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::BHBroadcastStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::BHBroadcastStage::_setsToMaterialize = nullptr;