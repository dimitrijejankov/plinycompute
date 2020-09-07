#include "TRAShuffleStage.h"

namespace pdb {

bool TRAShuffleStage::setup(const Handle<pdb::ExJob> &job,
                            const PDBPhysicalAlgorithmStatePtr &state,
                            const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                            const std::string &error) {
  return PDBPhysicalAlgorithmStage::setup(job, state, storage, error);
}

bool TRAShuffleStage::run(const Handle<pdb::ExJob> &job,
                          const PDBPhysicalAlgorithmStatePtr &state,
                          const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                          const std::string &error) {
  return PDBPhysicalAlgorithmStage::run(job, state, storage, error);
}

void TRAShuffleStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                              const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  PDBPhysicalAlgorithmStage::cleanup(state, storage);
}

TRAShuffleStage::TRAShuffleStage(const std::string &db, const std::string &set,
                                 const std::string &sink, const pdb::Vector<int32_t> &indices) :
     PDBPhysicalAlgorithmStage(*(_sink),
                               *(_sources),
                               *(_finalTupleSet),
                               *(_secondarySources),
                               *(_setsToMaterialize)), db(db), set(set), sink(sink) {}


}

const pdb::PDBSinkPageSetSpec *pdb::TRAShuffleStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::TRAShuffleStage::_sources = nullptr;
const pdb::String *pdb::TRAShuffleStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::TRAShuffleStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::TRAShuffleStage::_setsToMaterialize = nullptr;