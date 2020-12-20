#pragma once

#include <PDBSetObject.h>
#include "PDBPhysicalAlgorithmStage.h"

namespace pdb {

class PDBJoin3AggAsyncStage : public PDBPhysicalAlgorithmStage {
 public:

  PDBJoin3AggAsyncStage(const PDBSetObject &sourceSet0,
                        const PDBSetObject &sourceSet1,
                        const PDBSetObject &sourceSet2,
                        const pdb::String &in0,
                        const pdb::String &out0,
                        const pdb::String &in1,
                        const pdb::String &out1,
                        const pdb::String &in2,
                        const pdb::String &out2,
                        const pdb::String &out3,
                        const pdb::String &final) : sourceSet0(sourceSet0),
                                                    sourceSet1(sourceSet1),
                                                    sourceSet2(sourceSet2),
                                                    in0(in0),
                                                    in1(in1),
                                                    in2(in2),
                                                    out0(out0),
                                                    out1(out1),
                                                    out2(out2),
                                                    out3(out3),
                                                    final(final),
                                                    PDBPhysicalAlgorithmStage(PDBSinkPageSetSpec(),
                                                                              Vector<PDBSourceSpec>(),
                                                                              out0,
                                                                              Vector<pdb::Handle<PDBSourcePageSetSpec>>(),
                                                                              Vector<PDBSetObject>()) {

  }

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage);

  const PDBSetObject &sourceSet0;
  const PDBSetObject &sourceSet1;
  const PDBSetObject &sourceSet2;
  const pdb::String &in0;
  const pdb::String &in1;
  const pdb::String &in2;

  const pdb::String &out0;
  const pdb::String &out1;
  const pdb::String &out2;
  const pdb::String &out3;
  const pdb::String &final;

};

}