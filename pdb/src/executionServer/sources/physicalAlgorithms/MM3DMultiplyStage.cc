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
  return PDBPhysicalAlgorithmStage::setup(job, state, storage, error);
}

bool pdb::MM3DMultiplyStage::run(const pdb::Handle<pdb::ExJob> &job,
                                 const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                 const shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                 const string &error) {
  return PDBPhysicalAlgorithmStage::run(job, state, storage, error);
}

void pdb::MM3DMultiplyStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                     const shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  PDBPhysicalAlgorithmStage::cleanup(state, storage);
}
