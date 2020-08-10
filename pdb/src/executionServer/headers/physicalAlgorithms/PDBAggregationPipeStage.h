#pragma once

#include <PDBPhysicalAlgorithm.h>

namespace pdb {


class PDBAggregationPipeStage : public PDBPhysicalAlgorithmStage {
public:

  PDBAggregationPipeStage(const PDBSinkPageSetSpec &sink,
                          const Vector<PDBSourceSpec> &sources,
                          const String &final_tuple_set,
                          const Vector<Handle<PDBSourcePageSetSpec>> &secondary_sources,
                          const Vector<PDBSetObject> &sets_to_materialize,
                          const PDBSinkPageSetSpec &hashed_to_send,
                          const PDBSourcePageSetSpec &hashed_to_recv);

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManagerFrontend> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManagerFrontend> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerFrontend> &storage) override;

  // The sink tuple set where we are putting stuff
  const PDBSinkPageSetSpec &hashedToSend;

  // The sink type the algorithm should setup
  const PDBSourcePageSetSpec &hashedToRecv;

};


}