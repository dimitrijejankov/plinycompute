#pragma once

#include <PDBPhysicalAlgorithm.h>

namespace pdb {


class TRALocalJoinStage : public PDBPhysicalAlgorithmStage {
 public:

  TRALocalJoinStage(const std::string &lhsPageSet, const std::string &rhsDB, const std::string &rhsSet, const std::string &sink,
                    const pdb::Vector<int32_t> &lhsIndices, const pdb::Vector<int32_t> &rhsIndices,
                    const std::string &firstTupleSet, const std::string &finalTupleSet);

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
  pdb::String rhsDB;

  // source set
  pdb::String rhsSet;

  // the page
  pdb::String lhsPageSet;

  // sink
  pdb::String sink;

  // the first page set of the join pipeline
  pdb::String firstTupleSet;

  // the last page set of the join pipeline
  pdb::String finalTupleSet;

  Handle<Vector<Handle<Computation>>> computations;

  pdb::String TCAPString;

  const pdb::Vector<int32_t> &lhsIndices;
  const pdb::Vector<int32_t> &rhsIndices;

  const static PDBSinkPageSetSpec *_sink;
  const static Vector<PDBSourceSpec> *_sources;
  const static String *_finalTupleSet;
  const static Vector<pdb::Handle<PDBSourcePageSetSpec>> *_secondarySources;
  const static Vector<PDBSetObject> *_setsToMaterialize;
};

}