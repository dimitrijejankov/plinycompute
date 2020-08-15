#pragma once

#include "PDBPhysicalAlgorithm.h"
#include "PageProcessor.h"
#include "PDBPageNetworkSender.h"
#include "PDBPageSelfReceiver.h"
#include "PipelineInterface.h"
#include "Computation.h"
#include "PDBPhysicalAlgorithmStage.h"

// PRELOAD %PDBShuffleForJoinAlgorithm%

namespace pdb {

class PDBShuffleForJoinAlgorithm : public PDBPhysicalAlgorithm {
public:

  PDBShuffleForJoinAlgorithm() = default;

  PDBShuffleForJoinAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                             const AtomicComputationPtr &finalAtomicComputation,
                             const pdb::Handle<pdb::PDBSinkPageSetSpec> &intermediate,
                             const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                             const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                             const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize);

  ENABLE_DEEP_COPY

  /**
   * Returns the initial state for the shuffle for join stage
   * @return the initial state
   */
  [[nodiscard]] PDBPhysicalAlgorithmStatePtr getInitialState(const Handle<pdb::ExJob> &job, NodeConfigPtr config) const override;

  /**
   * Return the one stage of the shuffle for join algorithm
   * @return the one stage in a vector
   */
  [[nodiscard]] PDBPhysicalAlgorithmStagePtr getNextStage(const PDBPhysicalAlgorithmStatePtr &state) override;

  /**
   * Returns one since we only have one stage
   * @return 1
   */
  [[nodiscard]] int32_t numStages() const override;

  /**
   * Returns ShuffleForJoinAlgorithm as the type
   * @return the type
   */
  PDBPhysicalAlgorithmType getAlgorithmType() override;

 private:

  /**
   * The intermediate page set
   */
  pdb::Handle<PDBSinkPageSetSpec> intermediate;

  FRIEND_TEST(TestPhysicalOptimizer, TestJoin2);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin3);
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregationAfterTwoWayJoin);
};

}