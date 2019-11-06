#pragma once

#include <gtest/gtest_prod.h>
#include <PipelineInterface.h>
#include <processors/PreaggregationPageProcessor.h>
#include "PDBPhysicalAlgorithm.h"
#include "PDBPageSelfReceiver.h"
#include "Computation.h"
#include "PDBPageNetworkSender.h"

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
                              const AtomicComputationPtr& leftInputTupleSet,
                              const AtomicComputationPtr& rightInputTupleSet,
                              const AtomicComputationPtr& joinTupleSet,
                              const AtomicComputationPtr& aggregationKey,
                              pdb::Handle<PDBSinkPageSetSpec> &hashedLHSKey,
                              pdb::Handle<PDBSinkPageSetSpec> &hashedRHSKey,
                              pdb::Handle<PDBSinkPageSetSpec> &aggregationTID,
                              const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                              const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize);

  ~PDBJoinAggregationAlgorithm() override = default;

  /**
   * //TODO
   */
  bool setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) override;

  /**
   * //TODO
   */
  bool run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

  /**
   *
   */
  void cleanup() override;

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

  /**
   * The lhs input set to the join aggregation pipeline
   */
  pdb::String leftInputTupleSet;

  /**
   * The rhs input set to the join aggregation pipeline
   */
  pdb::String rightInputTupleSet;

  /**
   * The join tuple set
   */
  pdb::String joinTupleSet;

  /**
   * The sources of the right side of the merged pipeline
   */
  pdb::Vector<PDBSourceSpec> rightSources;

  /**
   * this page set is going to have the intermediate results of the LHS, the it is going to contain the JoinMap<hash, LHSKey>
   */
  pdb::Handle<PDBSinkPageSetSpec> hashedLHSKey;

  /**
   * this page set is going to have the intermediate results of the RHS, the it is going to contain the JoinMap<hash, RHSKey>
   */
  pdb::Handle<PDBSinkPageSetSpec> hashedRHSKey;

  /**
   * this page set is going to have the intermediate results of the Aggregation Keys, the it is going to contain the JoinMap<AGG_TID, Vector<pair<LHS_TID, RHS_TID>>
   * there are also going to be two anonymous pages with Map<LHSKey, LHS_TID> and Map<RHSKey, RHS_Key>.
   */
  pdb::Handle<PDBSinkPageSetSpec> aggregationTID;

  FRIEND_TEST(TestPhysicalOptimizer, TestKeyedMatrixMultipply);
};

}