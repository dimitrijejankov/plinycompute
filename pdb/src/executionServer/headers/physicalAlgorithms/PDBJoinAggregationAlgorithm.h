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
                              AtomicComputationPtr leftInputTupleSet,
                              AtomicComputationPtr rightInputTupleSet,
                              AtomicComputationPtr joinTupleSet,
                              AtomicComputationPtr aggregationKey,
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
  AtomicComputationPtr leftInputTupleSet;

  /**
   * The rhs input set to the join aggregation pipeline
   */
  AtomicComputationPtr rightInputTupleSet;

  /**
   * The join tuple set
   */
  AtomicComputationPtr joinTupleSet;

  /**
   * The the key extraction tuple set of the aggregation
   */
  AtomicComputationPtr aggregationKey;

  /**
   * The sources of the right side of the merged pipeline
   */
  pdb::Vector<PDBSourceSpec> rightSources;

  FRIEND_TEST(TestPhysicalOptimizer, TestKeyedMatrixMultipply);
};

}