#pragma once

#include "PDBPhysicalAlgorithmStage.h"
#include "PDBPageNetworkSender.h"
#include "PDBPageSelfReceiver.h"
#include "PDBJoinAggregationState.h"
#include <physicalAlgorithms/PDBJoin3AlgorithmState.h>

namespace pdb {

class PDBJoin3AlgorithmKeyStage : public PDBPhysicalAlgorithmStage {
 public:

  PDBJoin3AlgorithmKeyStage(const PDBSetObject &sourceSet0,
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

  PDBAbstractPageSetPtr getKeySourcePageSet(const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                            const pdb::Vector<PDBSourceSpec> &srcs);

  PDBAbstractPageSetPtr getFetchingPageSet(const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                           const pdb::Vector<PDBSourceSpec> &srcs,
                                           const std::string &ip,
                                           int32_t port);

  static bool setupSenders(const Handle<pdb::ExJob> &job,
                           const std::shared_ptr<PDBJoin3AlgorithmState> &state,
                           const PDBSourcePageSetSpec &recvPageSet,
                           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                           std::shared_ptr<std::vector<PDBPageQueuePtr>> &pageQueues,
                           std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> &senders,
                           PDBPageSelfReceiverPtr *selfReceiver);

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) override;

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

 private:

  pdb::SourceSetArgPtr getKeySourceSetArg(std::shared_ptr<pdb::PDBCatalogClient> &catalogClient);

  static bool runLead(const Handle<pdb::ExJob> &job,
                      const PDBPhysicalAlgorithmStatePtr &state,
                      const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                      const std::string &error);

  static bool runFollower(const Handle<pdb::ExJob> &job,
                          const PDBPhysicalAlgorithmStatePtr &state,
                          const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                          const std::string &error);
};

}