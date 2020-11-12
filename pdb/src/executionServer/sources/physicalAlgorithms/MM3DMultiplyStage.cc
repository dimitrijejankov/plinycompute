#include "MM3DMultiplyStage.h"

pdb::MM3DMultiplyStage::MM3DMultiplyStage() :
    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)) {}

bool pdb::MM3DMultiplyStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                   const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                   const shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                   const string &error) {
  return true;
}

bool pdb::MM3DMultiplyStage::run(const pdb::Handle<pdb::ExJob> &job,
                                 const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                 const shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                 const string &error) {
  return true;
}

void pdb::MM3DMultiplyStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                     const shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
}

const pdb::PDBSinkPageSetSpec *pdb::MM3DMultiplyStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::MM3DMultiplyStage::_sources = nullptr;
const pdb::String *pdb::MM3DMultiplyStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::MM3DMultiplyStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::MM3DMultiplyStage::_setsToMaterialize = nullptr;