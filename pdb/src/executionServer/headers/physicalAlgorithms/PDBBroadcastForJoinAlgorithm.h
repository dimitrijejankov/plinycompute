#pragma once

#include "PDBPhysicalAlgorithm.h"
#include <ComputePlan.h>
#include <PDBCatalogClient.h>
#include <ExJob.h>
#include <PDBStorageManagerBackend.h>
#include <PDBPageNetworkSender.h>
#include <BroadcastJoinProcessor.h>
#include <PDBPageSelfReceiver.h>
#include <GenericWork.h>
#include <memory>


// PRELOAD %PDBBroadcastForJoinAlgorithm%

namespace pdb {

class PDBBroadcastForJoinAlgorithm : public PDBPhysicalAlgorithm {
 public:

  PDBBroadcastForJoinAlgorithm() = default;

  PDBBroadcastForJoinAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                               const AtomicComputationPtr &finalAtomicComputation,
                               const pdb::Handle<pdb::PDBSinkPageSetSpec> &hashedToSend,
                               const pdb::Handle<pdb::PDBSourcePageSetSpec> &hashedToRecv,
                               const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                               const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                               const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize);

  ENABLE_DEEP_COPY

  /**
   * Returns DistributedAggregation as the type
   * @return the type
   */
  PDBPhysicalAlgorithmType getAlgorithmType() override;

  [[nodiscard]] PDBPhysicalAlgorithmStatePtr getInitialState(const pdb::Handle<pdb::ExJob> &job) const override;

  [[nodiscard]] vector<PDBPhysicalAlgorithmStagePtr> getStages() const override;

  [[nodiscard]] int32_t numStages() const override;

 private:

  /**
   * The sink tuple set where we are putting stuff
   */
  pdb::Handle<PDBSinkPageSetSpec> hashedToSend;

  /**
   * The sink type the algorithm should setup
   */
  pdb::Handle<PDBSourcePageSetSpec> hashedToRecv;

  // mark the tests that are testing this algorithm
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin1);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin3);
};

}