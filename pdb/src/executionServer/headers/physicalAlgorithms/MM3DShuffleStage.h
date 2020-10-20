#pragma once

#include <PDBPhysicalAlgorithm.h>

namespace pdb {


class MM3DShuffleStage : public PDBPhysicalAlgorithmStage {
 public:

  MM3DShuffleStage();

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

  const static PDBSinkPageSetSpec *_sink;
  const static Vector<PDBSourceSpec> *_sources;
  const static String *_finalTupleSet;
  const static Vector<pdb::Handle<PDBSourcePageSetSpec>> *_secondarySources;
  const static Vector<PDBSetObject> *_setsToMaterialize;
};


}