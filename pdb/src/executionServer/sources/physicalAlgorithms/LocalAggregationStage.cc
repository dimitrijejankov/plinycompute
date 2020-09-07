#include "LocalAggregationStage.h"

namespace pdb {

bool LocalAggregationStage::setup(const Handle<pdb::ExJob> &job,
                           const PDBPhysicalAlgorithmStatePtr &state,
                           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                           const std::string &error) {
  return PDBPhysicalAlgorithmStage::setup(job, state, storage, error);
}

bool LocalAggregationStage::run(const Handle<pdb::ExJob> &job,
                         const PDBPhysicalAlgorithmStatePtr &state,
                         const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                         const std::string &error) {
  return PDBPhysicalAlgorithmStage::run(job, state, storage, error);
}

void LocalAggregationStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  PDBPhysicalAlgorithmStage::cleanup(state, storage);
}

LocalAggregationStage::LocalAggregationStage(const std::string &db, const std::string &set,
                               const std::string &sink, const pdb::Vector<int32_t> &indices) :
    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)), db(db), set(set), sink(sink) {}


}

const pdb::PDBSinkPageSetSpec *pdb::LocalAggregationStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::LocalAggregationStage::_sources = nullptr;
const pdb::String *pdb::LocalAggregationStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::LocalAggregationStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::LocalAggregationStage::_setsToMaterialize = nullptr;