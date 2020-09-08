#include "TRALocalJoinStage.h"

namespace pdb {

bool TRALocalJoinStage::setup(const Handle<pdb::ExJob> &job,
                              const PDBPhysicalAlgorithmStatePtr &state,
                              const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                              const std::string &error) {
  std::cout << "setup\n";
  return true;
}

bool TRALocalJoinStage::run(const Handle<pdb::ExJob> &job,
                            const PDBPhysicalAlgorithmStatePtr &state,
                            const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                            const std::string &error) {
  std::cout << "run\n";
  return true;
}

void TRALocalJoinStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  std::cout << "Cleanup\n";
}

TRALocalJoinStage::TRALocalJoinStage(const std::string &db, const std::string &set,
                                     const std::string &sink, const pdb::Vector<int32_t> &indices) :
    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)), db(db), set(set), sink(sink) {}


}

const pdb::PDBSinkPageSetSpec *pdb::TRALocalJoinStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::TRALocalJoinStage::_sources = nullptr;
const pdb::String *pdb::TRALocalJoinStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::TRALocalJoinStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::TRALocalJoinStage::_setsToMaterialize = nullptr;