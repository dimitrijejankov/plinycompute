#pragma once

#include <PDBPhysicalAlgorithm.h>
#include <PDBCatalogClient.h>

namespace pdb {


class PDBStraightPipeStage : public PDBPhysicalAlgorithmStage {
public:

  PDBStraightPipeStage(const PDBSinkPageSetSpec &sink,
                       const Vector<PDBSourceSpec> &sources,
                       const String &finalTupleSet,
                       const Vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                       const Vector<PDBSetObject> &setsToMaterialize);

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManager> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManager> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManager> &storage) override;

};


}