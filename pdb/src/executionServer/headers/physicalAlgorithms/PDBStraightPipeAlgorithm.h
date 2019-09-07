//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_STRAIGHTPIPEALGORITHM_H
#define PDB_STRAIGHTPIPEALGORITHM_H

#include "PDBStorageManagerBackend.h"
#include "PDBPhysicalAlgorithm.h"
#include "Computation.h"
#include "pipeline/Pipeline.h"
#include <vector>

/**
 * This is important do not remove, it is used by the generator
 */

namespace pdb {

// PRELOAD %PDBStraightPipeAlgorithm%

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
   * Vector of pipelines that will run this algorithm. The pipelines will be built when you call setup on this object.
   * This must be null when sending this object.
   */
  std::shared_ptr<std::vector<PipelinePtr>> myPipelines = nullptr;


  FRIEND_TEST(TestPhysicalOptimizer, TestJoin3);
  FRIEND_TEST(TestPhysicalOptimizer, TestTwoSinksSelection);
  FRIEND_TEST(TestPhysicalOptimizer, TestUnion1);
  FRIEND_TEST(TestPhysicalOptimizer, TestUnion2);
};

}


#endif //PDB_STRAIGHTPIPEALGORITHM_H
