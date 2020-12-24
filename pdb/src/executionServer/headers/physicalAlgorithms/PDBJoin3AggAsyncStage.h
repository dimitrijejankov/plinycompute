#pragma once

#include <PDBSetObject.h>
#include "PDBPhysicalAlgorithmStage.h"
#include "JoinPlannerResult.h"

namespace pdb {

struct meta_t {

  int32_t rowID;
  int32_t colID;
  int32_t numRows;
  int32_t numCols;
  bool hasMore;
};

struct emitter_row_t {

  // the row and col id of the projected join record
  int32_t rowID = -1;
  int32_t colID = -1;

  // pointer to the data
  void *a = nullptr;
  void *b = nullptr;
  void *c = nullptr;
};


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

  void setup_set_comm(const std::string &set,
                      std::mutex &m,
                      std::condition_variable &cv,
                      atomic_int &counter,
                      std::vector<int32_t> &joined,
                      std::vector<std::vector<int32_t>> &records_to_join,
                      std::vector<emitter_row_t> &to_join,
                      pdb::Handle<JoinPlannerResult> &plan,
                      const pdb::Handle<pdb::ExJob> &job,
                      const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                      const pdb::PDBPhysicalAlgorithmStatePtr &state,
                      PDBBuzzerPtr &tempBuzzer);

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

  std::vector<emitter_row_t> toEmmit;
};

}