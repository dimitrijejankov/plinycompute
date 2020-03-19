#pragma once

#include <PDBSetObject.h>
#include "PDBPhysicalAlgorithmStage.h"

namespace pdb {

class PDBJoin8AlgorithmJoinStage : public PDBPhysicalAlgorithmStage {
public:

  PDBJoin8AlgorithmJoinStage(const PDBSetObject &sourceSet,
                             const PDBSetObject &sinkSet,
                             const pdb::String &in0,
                             const pdb::String &out0,
                             const pdb::String &in1,
                             const pdb::String &out1,
                             const pdb::String &in2,
                             const pdb::String &out2,
                             const pdb::String &in3,
                             const pdb::String &out3,
                             const pdb::String &in4,
                             const pdb::String &out4,
                             const pdb::String &in5,
                             const pdb::String &out5,
                             const pdb::String &in6,
                             const pdb::String &out6,
                             const pdb::String &in7,
                             const pdb::String &out7) : sourceSet(sourceSet),
                                                        sinkSet(sinkSet),
                                                        in0(in0),
                                                        in1(in1),
                                                        in2(in2),
                                                        in3(in3),
                                                        in4(in4),
                                                        in5(in5),
                                                        in6(in6),
                                                        in7(in7),
                                                        out0(out0),
                                                        out1(out1),
                                                        out2(out2),
                                                        out3(out3),
                                                        out4(out4),
                                                        out5(out5),
                                                        out6(out6),
                                                        out7(out7),
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

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) override;

  const PDBSetObject &sourceSet;
  const PDBSetObject &sinkSet;
  const pdb::String &in0;
  const pdb::String &out0;
  const pdb::String &in1;
  const pdb::String &out1;
  const pdb::String &in2;
  const pdb::String &out2;
  const pdb::String &in3;
  const pdb::String &out3;
  const pdb::String &in4;
  const pdb::String &out4;
  const pdb::String &in5;
  const pdb::String &out5;
  const pdb::String &in6;
  const pdb::String &out6;
  const pdb::String &in7;
  const pdb::String &out7;
};

}