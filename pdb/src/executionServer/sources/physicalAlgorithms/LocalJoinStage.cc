#include "LocalJoinStage.h"

namespace pdb {

bool LocalJoinStage::setup(const Handle<pdb::ExJob> &job,
                           const PDBPhysicalAlgorithmStatePtr &state,
                           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                           const std::string &error) {
  return PDBPhysicalAlgorithmStage::setup(job, state, storage, error);
}

bool LocalJoinStage::run(const Handle<pdb::ExJob> &job,
                         const PDBPhysicalAlgorithmStatePtr &state,
                         const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                         const std::string &error) {
  return PDBPhysicalAlgorithmStage::run(job, state, storage, error);
}

void LocalJoinStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  PDBPhysicalAlgorithmStage::cleanup(state, storage);
}

LocalJoinStage::LocalJoinStage(const std::string &db, const std::string &set,
                               const std::string &sink, const pdb::Vector<int32_t> &indices) :
    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)), db(db), set(set), sink(sink) {}


}

const pdb::PDBSinkPageSetSpec *pdb::LocalJoinStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::LocalJoinStage::_sources = nullptr;
const pdb::String *pdb::LocalJoinStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::LocalJoinStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::LocalJoinStage::_setsToMaterialize = nullptr;