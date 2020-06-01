#pragma once

#include "PDBPhysicalAlgorithmStage.h"

namespace pdb {

class PDBJoinAggregationAggregationStage : public PDBPhysicalAlgorithmStage {
 public:

  PDBJoinAggregationAggregationStage(const PDBSinkPageSetSpec &sink,
                                     const PDBSinkPageSetSpec &preaggIntermediate,
                                     const Vector<PDBSourceSpec> &sources,
                                     const String &final_tuple_set,
                                     const Vector<pdb::Handle<PDBSourcePageSetSpec>> &secondary_sources,
                                     const Vector<PDBSetObject> &sets_to_materialize,
                                     const String &join_tuple_set);

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

 private:

  // the join tuple set
  const pdb::String &joinTupleSet;

  //
  const PDBSinkPageSetSpec &preaggIntermediate;
};

}