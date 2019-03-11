//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_STRAIGHTPIPEALGORITHM_H
#define PDB_STRAIGHTPIPEALGORITHM_H

#include "PDBStorageManagerBackend.h"
#include "PDBPhysicalAlgorithm.h"
#include "Pipeline.h"

/**
 * This is important do not remove, it is used by the generator
 */

namespace pdb {

// PRELOAD %PDBStraightPipeAlgorithm%

class PDBStraightPipeAlgorithm : public PDBPhysicalAlgorithm {
public:

  ENABLE_DEEP_COPY

  PDBStraightPipeAlgorithm() = default;
  ~PDBStraightPipeAlgorithm() = default;


  PDBStraightPipeAlgorithm(const pdb::Handle<PDBSourcePageSetSpec> &source,
                           const pdb::Handle<PDBSinkPageSetSpec> &sink,
                           const pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> &secondarySources);

  /**
   * //TODO
   */
  bool setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) override;

  /**
   * //TODO
   */
  bool run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

  /**
   * Returns StraightPipe as the type
   * @return the type
   */
  PDBPhysicalAlgorithmType getAlgorithmType() override;

private:

  /**
   * The pipeline this thing is going to run, it is initialized when you call setup on this object
   */
  PipelinePtr myPipeline;

  /**
   * The name of the database <databaseName, setName> initialized when you call setup on this object, this has to be null when sending
   * meaning once you run the algorithm the algorithm can not go over the wire!
   */
  std::shared_ptr<std::pair<std::string, std::string>> outputSet = nullptr;


  /**
   * Should we materialize this or not?
   */
  bool shouldMaterialize = false;

};

}


#endif //PDB_STRAIGHTPIPEALGORITHM_H
