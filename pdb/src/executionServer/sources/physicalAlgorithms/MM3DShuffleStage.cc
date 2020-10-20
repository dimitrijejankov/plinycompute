#include "MM3DShuffleStage.h"

pdb::MM3DShuffleStage::MM3DShuffleStage() :

    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)) {}
bool pdb::MM3DShuffleStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                  const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                  const shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                  const string &error) {
  return PDBPhysicalAlgorithmStage::setup(job, state, storage, error);
}

bool pdb::MM3DShuffleStage::run(const pdb::Handle<pdb::ExJob> &job,
                                const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                const shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                const string &error) {
  return PDBPhysicalAlgorithmStage::run(job, state, storage, error);
}

void pdb::MM3DShuffleStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                    const shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  PDBPhysicalAlgorithmStage::cleanup(state, storage);
}