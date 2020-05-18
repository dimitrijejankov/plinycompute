#pragma once

#include <gtest/gtest_prod.h>
#include <PipelineInterface.h>
#include <processors/PreaggregationPageProcessor.h>
#include "PDBPhysicalAlgorithm.h"
#include "PDBPageSelfReceiver.h"
#include "Computation.h"
#include "PDBPageNetworkSender.h"
#include "PDBAnonymousPageSet.h"
#include "JoinAggPlanner.h"
#include "PDBLabeledPageSet.h"
#include "JoinAggSideSender.h"
#include "JoinMapCreator.h"

namespace pdb {

// PRELOAD %PDBJoinAggregationAlgorithm%

/**
 * Basically executes a pipeline that looks like this :
 *
 *        agg
 *         |
 *        join
 *       /    \
 *     lhs   rhs
 *
 * This algorithm should only be use for cases where there is a few records that are very large
 */
class PDBJoinAggregationAlgorithm : public PDBPhysicalAlgorithm {
 public:

  ENABLE_DEEP_COPY

  PDBJoinAggregationAlgorithm() = default;

  PDBJoinAggregationAlgorithm(const std::vector<PDBPrimarySource> &leftSource,
                              const std::vector<PDBPrimarySource> &rightSource,
                              const pdb::Handle<PDBSinkPageSetSpec> &sink,
                              const pdb::Handle<PDBSinkPageSetSpec> &leftKeySink,
                              const pdb::Handle<PDBSinkPageSetSpec> &rightKeySink,
                              const pdb::Handle<PDBSinkPageSetSpec> &joinAggKeySink,
                              const pdb::Handle<PDBSinkPageSetSpec> &intermediateSink,
                              const pdb::Handle<PDBSinkPageSetSpec> &preaggIntermediate,
                              const pdb::Handle<PDBSourcePageSetSpec> &leftKeySource,
                              const pdb::Handle<PDBSourcePageSetSpec> &rightKeySource,
                              const pdb::Handle<PDBSourcePageSetSpec> &leftJoinSource,
                              const pdb::Handle<PDBSourcePageSetSpec> &rightJoinSource,
                              const pdb::Handle<PDBSourcePageSetSpec> &planSource,
                              const AtomicComputationPtr& leftInputTupleSet,
                              const AtomicComputationPtr& rightInputTupleSet,
                              const AtomicComputationPtr& joinTupleSet,
                              const AtomicComputationPtr& aggregationKey,
                              const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                              const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize);

  ~PDBJoinAggregationAlgorithm() override = default;

  [[nodiscard]] PDBPhysicalAlgorithmStatePtr getInitialState(const pdb::Handle<pdb::ExJob> &job) const override;

  [[nodiscard]] PDBPhysicalAlgorithmStagePtr getNextStage(const PDBPhysicalAlgorithmStatePtr &state) override;

  [[nodiscard]] int32_t numStages() const override;

  /**
   * Returns StraightPipe as the type
   * @return the type
   */
  PDBPhysicalAlgorithmType getAlgorithmType() override;

  /**
   * The output container type of the straight pipeline is always a vector, meaning the root object is always a pdb::Vector
   * @return PDB_CATALOG_SET_VECTOR_CONTAINER
   */
  PDBCatalogSetContainerType getOutputContainerType() override;

 private:

  // The lhs input set to the join aggregation pipeline
  pdb::String leftInputTupleSet;

  // The rhs input set to the join aggregation pipeline
  pdb::String rightInputTupleSet;

  // The join tuple set
  pdb::String joinTupleSet;

  // the page set we use to store the result of the left key pipeline
  Handle<PDBSinkPageSetSpec> lhsKeySink;

  // the page set we use to store the result of the right key pipeline
  Handle<PDBSinkPageSetSpec> rhsKeySink;

  // the final sink where we store the keys of the aggregation
  Handle<PDBSinkPageSetSpec> joinAggKeySink;

  // The sources of the right side of the merged pipeline
  pdb::Vector<PDBSourceSpec> rightSources;

  // this page set is going to have the intermediate results of the LHS, the it is going to contain the JoinMap<hash, LHSKey>
  pdb::Handle<PDBSinkPageSetSpec> hashedLHSKey;

  // this page set is going to have the intermediate results of the RHS, the it is going to contain the JoinMap<hash, RHSKey>
  pdb::Handle<PDBSinkPageSetSpec> hashedRHSKey;

   // this page set is going to have the intermediate results of the Aggregation Keys, the it is going to contain the JoinMap<AGG_TID, Vector<pair<LHS_TID, RHS_TID>>
   // there are also going to be two anonymous pages with Map<LHSKey, LHS_TID> and Map<RHSKey, RHS_Key>.
  pdb::Handle<PDBSinkPageSetSpec> aggregationTID;

  /**
   *
   */
  pdb::Handle<PDBSinkPageSetSpec> intermediateSink;

  /**
   *
   */
  pdb::Handle<PDBSinkPageSetSpec> preaggIntermediate;

  /**
   *
   */
  pdb::Handle<PDBSourcePageSetSpec> leftJoinSource;

  /**
   *
   */
  pdb::Handle<PDBSourcePageSetSpec> rightJoinSource;

  /**
 *
 */
  pdb::Handle<PDBSourcePageSetSpec> leftKeySource;

  /**
   *
   */
  pdb::Handle<PDBSourcePageSetSpec> rightKeySource;

  /**
   *
   */
  pdb::Handle<PDBSourcePageSetSpec> planSource;


  FRIEND_TEST(TestPhysicalOptimizer, TestKeyedMatrixMultipply);
};

}