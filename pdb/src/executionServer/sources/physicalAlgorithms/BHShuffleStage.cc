#include "BHShuffleStage.h"

namespace pdb {

bool BHShuffleStage::setup(const Handle<pdb::ExJob> &job,
                           const PDBPhysicalAlgorithmStatePtr &state,
                           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                           const std::string &error) {
  return PDBPhysicalAlgorithmStage::setup(job, state, storage, error);
}

bool BHShuffleStage::run(const Handle<pdb::ExJob> &job,
                         const PDBPhysicalAlgorithmStatePtr &state,
                         const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                         const std::string &error) {
  return PDBPhysicalAlgorithmStage::run(job, state, storage, error);
}

void BHShuffleStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  PDBPhysicalAlgorithmStage::cleanup(state, storage);
}

BHShuffleStage::BHShuffleStage(const std::string &db, const std::string &set,
                               const std::string &sink, const pdb::Vector<int32_t> &indices) :
     PDBPhysicalAlgorithmStage(*(_sink),
                               *(_sources),
                               *(_finalTupleSet),
                               *(_secondarySources),
                               *(_setsToMaterialize)), db(db), set(set), sink(sink) {}


}

const pdb::PDBSinkPageSetSpec *pdb::BHShuffleStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::BHShuffleStage::_sources = nullptr;
const pdb::String *pdb::BHShuffleStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::BHShuffleStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::BHShuffleStage::_setsToMaterialize = nullptr;