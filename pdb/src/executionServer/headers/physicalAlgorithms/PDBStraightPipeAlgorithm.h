//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_STRAIGHTPIPEALGORITHM_H
#define PDB_STRAIGHTPIPEALGORITHM_H

#include "PDBStorageManagerBackend.h"
#include "PDBPhysicalAlgorithm.h"
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

  PDBStraightPipeAlgorithm(const std::string &firstTupleSet,
                           const std::string &finalTupleSet,
                           const pdb::Handle<PDBSourcePageSetSpec> &source,
                           const pdb::Handle<PDBSinkPageSetSpec> &sink,
                           const pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &secondarySources);

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

private:

  /**
   * Vector of pipelines that will run this algorithm. The pipelines will be built when you call setup on this object.
   * This must be null when sending this object.
   */
  std::shared_ptr<std::vector<PipelinePtr>> myPipelines = nullptr;

  /**
   * The name of the database <databaseName, setName> initialized when you call setup on this object, this has to be null when sending
   * meaning once you run the algorithm the algorithm can not go over the wire!
   */
  std::shared_ptr<std::pair<std::string, std::string>> outputSet = nullptr;

  /**
   * Should we materialize this or not?
   */
  bool shouldMaterialize = false;

  FRIEND_TEST(TestPhysicalOptimizer, TestJoin3);
};

}


#endif //PDB_STRAIGHTPIPEALGORITHM_H
