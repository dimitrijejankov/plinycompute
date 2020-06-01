#pragma once

#include <PDBPhysicalAlgorithm.h>

namespace pdb {


class PDBBroadcastForJoinStage : public PDBPhysicalAlgorithmStage {
public:

  PDBBroadcastForJoinStage(const PDBSinkPageSetSpec &sink,
                           const Vector<PDBSourceSpec> &sources,
                           const String &final_tuple_set,
                           const Vector<Handle<PDBSourcePageSetSpec>> &secondary_sources,
                           const Vector<PDBSetObject> &sets_to_materialize,
                           PDBSinkPageSetSpec &hashed_to_send,
                           PDBSourcePageSetSpec &hashed_to_recv);

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

  // The sink tuple set where we are putting stuff
  PDBSinkPageSetSpec &hashedToSend;

  // The sink tuple set where we are putting stuff
  PDBSourcePageSetSpec &hashedToRecv;
};


}