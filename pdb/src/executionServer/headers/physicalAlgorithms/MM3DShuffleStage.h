#pragma once

#include <TRABlock.h>
#include <PDBPhysicalAlgorithm.h>
#include "MM3DIdx.h"

namespace pdb {


class MM3DShuffleStage : public PDBPhysicalAlgorithmStage {
 public:

  MM3DShuffleStage(int32_t n, int32_t num_nodes, int32_t num_threads);

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

  MM3DIdx idx;

  std::vector<std::unordered_set<Handle<TRABlock>*>> to_send_lhs;
  std::vector<std::unordered_set<Handle<TRABlock>*>> to_send_rhs;

  const static PDBSinkPageSetSpec *_sink;
  const static Vector<PDBSourceSpec> *_sources;
  const static String *_finalTupleSet;
  const static Vector<pdb::Handle<PDBSourcePageSetSpec>> *_secondarySources;
  const static Vector<PDBSetObject> *_setsToMaterialize;
};


}