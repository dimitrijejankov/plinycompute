#pragma once

// PRELOAD %PDBAggregationPipeAlgorithm%

#include <gtest/gtest_prod.h>
#include <PipelineInterface.h>
#include <processors/PreaggregationPageProcessor.h>
#include "PDBPhysicalAlgorithm.h"
#include "PDBPageSelfReceiver.h"
#include "Computation.h"
#include "PDBPageNetworkSender.h"

namespace pdb {

class PDBAggregationPipeAlgorithm : public PDBPhysicalAlgorithm {
public:

  ENABLE_DEEP_COPY

  PDBAggregationPipeAlgorithm() = default;

  ~PDBAggregationPipeAlgorithm() override = default;

  PDBAggregationPipeAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                              const AtomicComputationPtr &finalAtomicComputation,
                              const Handle<PDBSinkPageSetSpec> &hashedToSend,
                              const Handle<PDBSourcePageSetSpec> &hashedToRecv,
                              const Handle<PDBSinkPageSetSpec> &sink,
                              const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                              const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize);

  /**
   *
   * @param job
   * @return
   */
  [[nodiscard]] PDBPhysicalAlgorithmStatePtr getInitialState(const pdb::Handle<pdb::ExJob> &job) const override;

  /**
   *
   * @return
   */
  [[nodiscard]] vector<PDBPhysicalAlgorithmStagePtr> getStages() const override;

  /**
   *
   * @return
   */
  [[nodiscard]] int32_t numStages() const override;

  /**
   * Returns BroadcastForJoin as the type
   * @return the type
   */
  PDBPhysicalAlgorithmType getAlgorithmType() override;

  /**
   * The pages of the aggregation always have as the root object pdb::map so it returns PDB_CATALOG_SET_MAP_CONTAINER
   * @return PDB_CATALOG_SET_MAP_CONTAINER
   */
  PDBCatalogSetContainerType getOutputContainerType() override;

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
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregation);
  FRIEND_TEST(TestPhysicalOptimizer, TestMultiSink);
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregationAfterTwoWayJoin);
};

}