#pragma once

#include <PDBPhysicalAlgorithm.h>

namespace pdb {


class TRAShuffleReplicateStage : public PDBPhysicalAlgorithmStage {
 public:

  TRAShuffleReplicateStage(const std::string &inputPageSet, int32_t newIdx, int32_t numRepl,
                           const std::string &sink, const pdb::Vector<int32_t> &indices);

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

  // the new index
  int32_t newIdx;

  // the number we need to replicate
  int32_t numRepl;

  // source db
  pdb::String db;

  // source set
  pdb::String set;

  // the page
  pdb::String inputPageSet;

  // sink
  pdb::String sink;

  //
  const pdb::Vector<int32_t> &indices;

  const static PDBSinkPageSetSpec *_sink;
  const static Vector<PDBSourceSpec> *_sources;
  const static String *_finalTupleSet;
  const static Vector<pdb::Handle<PDBSourcePageSetSpec>> *_secondarySources;
  const static Vector<PDBSetObject> *_setsToMaterialize;
};

}