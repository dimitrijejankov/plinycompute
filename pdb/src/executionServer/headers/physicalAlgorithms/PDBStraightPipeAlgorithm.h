#pragma once

#include "PDBPhysicalAlgorithm.h"
#include "PDBStraightPipeStage.h"
#include "Computation.h"
#include "pipeline/Pipeline.h"
#include <vector>

/**
 * This is important do not remove, it is used by the generator
 */

namespace pdb {

// PRELOAD %PDBStraightPipeAlgorithm%

/**
 * The straight pipeline has only one stage. That runs a pipeline.
 */
class PDBStraightPipeAlgorithm : public PDBPhysicalAlgorithm {
public:

  ENABLE_DEEP_COPY

  PDBStraightPipeAlgorithm() = default;

  ~PDBStraightPipeAlgorithm() override = default;

  PDBStraightPipeAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                           const AtomicComputationPtr &finalAtomicComputation,
                           const pdb::Handle<PDBSinkPageSetSpec> &sink,
                           const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                           const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize);

  /**
   * Returns the initial state for the stages of the straight pipeline
   * @return the initial state
   */
  [[nodiscard]] PDBPhysicalAlgorithmStatePtr getInitialState(const Handle<pdb::ExJob> &job, NodeConfigPtr config) const override;

  /**
   * Return the one stage of the straight pipeline algorithm
   * @return the one stage in a vector
   */
  [[nodiscard]] PDBPhysicalAlgorithmStagePtr getNextStage(const PDBPhysicalAlgorithmStatePtr &state) override;

  /**
   * Returns one since we only have one stage
   * @return 1
   */
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


  FRIEND_TEST(TestPhysicalOptimizer, TestJoin3);
  FRIEND_TEST(TestPhysicalOptimizer, TestTwoSinksSelection);
  FRIEND_TEST(TestPhysicalOptimizer, TestUnion1);
  FRIEND_TEST(TestPhysicalOptimizer, TestUnion2);
};

}