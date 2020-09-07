#pragma once

#include <PDBPhysicalAlgorithm.h>

namespace pdb {


class TRALocalAggregationStage : public PDBPhysicalAlgorithmStage {
 public:

  TRALocalAggregationStage(const std::string &db, const std::string &set, const std::string &sink, const pdb::Vector<int32_t> &indices);

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
  pdb::String inputPageSet;

  // sink
  pdb::String sink;

  const static PDBSinkPageSetSpec *_sink;
  const static Vector<PDBSourceSpec> *_sources;
  const static String *_finalTupleSet;
  const static Vector<pdb::Handle<PDBSourcePageSetSpec>> *_secondarySources;
  const static Vector<PDBSetObject> *_setsToMaterialize;
};

}