#pragma once

#include <PDBPhysicalAlgorithm.h>

namespace pdb {


class TRALocalAggregationStage : public PDBPhysicalAlgorithmStage {
 public:

  TRALocalAggregationStage(const pdb::String &inputPageSet,
                           const pdb::Vector<int32_t>& indices,
                           const pdb::String &sink);

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

  // source db
  pdb::String db;

  // source set
  pdb::String set;

  // the page
  const pdb::String &inputPageSet;

  // sink
  const pdb::String &sink;

  const pdb::Vector<int32_t>& indices;

  const static PDBSinkPageSetSpec *_sink;
  const static Vector<PDBSourceSpec> *_sources;
  const static String *_finalTupleSet;
  const static Vector<pdb::Handle<PDBSourcePageSetSpec>> *_secondarySources;
  const static Vector<PDBSetObject> *_setsToMaterialize;
};

}