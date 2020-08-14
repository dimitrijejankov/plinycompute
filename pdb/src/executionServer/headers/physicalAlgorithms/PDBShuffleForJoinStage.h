#pragma once

#include <PDBPhysicalAlgorithm.h>

namespace pdb {


class PDBShuffleForJoinStage : public PDBPhysicalAlgorithmStage {
 public:

  PDBShuffleForJoinStage(const PDBSinkPageSetSpec &sink,
                         const Vector<PDBSourceSpec> &sources,
                         const String &finalTupleSet,
                         const Vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                         const Vector<PDBSetObject> &setsToMaterialize,
                         const PDBSinkPageSetSpec &intermediate);

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManager> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManager> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManager> &storage) override;

  // page set identifier for the intermediate data created by the shuffle
  const PDBSinkPageSetSpec &intermediate;
};


}