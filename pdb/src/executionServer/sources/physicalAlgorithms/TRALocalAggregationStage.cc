#include "TRALocalAggregationStage.h"

namespace pdb {

bool TRALocalAggregationStage::setup(const Handle<pdb::ExJob> &job,
                                     const PDBPhysicalAlgorithmStatePtr &state,
                                     const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                     const std::string &error) {
  return PDBPhysicalAlgorithmStage::setup(job, state, storage, error);
}

bool TRALocalAggregationStage::run(const Handle<pdb::ExJob> &job,
                                   const PDBPhysicalAlgorithmStatePtr &state,
                                   const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                   const std::string &error) {
  return PDBPhysicalAlgorithmStage::run(job, state, storage, error);
}

void TRALocalAggregationStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                       const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  PDBPhysicalAlgorithmStage::cleanup(state, storage);
}

TRALocalAggregationStage::TRALocalAggregationStage(const std::string &db, const std::string &set,
                                                   const std::string &sink, const pdb::Vector<int32_t> &indices) :
    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)), db(db), set(set), sink(sink) {}


}

const pdb::PDBSinkPageSetSpec *pdb::TRALocalAggregationStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::TRALocalAggregationStage::_sources = nullptr;
const pdb::String *pdb::TRALocalAggregationStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::TRALocalAggregationStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::TRALocalAggregationStage::_setsToMaterialize = nullptr;