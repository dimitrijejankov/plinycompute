//
// Created by dimitrije on 3/20/19.
//

#ifndef PDB_PDBAGGREGATIONPIPEALGORITHM_H
#define PDB_PDBAGGREGATIONPIPEALGORITHM_H

// PRELOAD %PDBAggregationPipeAlgorithm%

#include "PDBPhysicalAlgorithm.h"
#include "Pipeline.h"
#include <vector>

namespace pdb {

class PDBAggregationPipeAlgorithm : public PDBPhysicalAlgorithm {
public:

  ENABLE_DEEP_COPY

  PDBAggregationPipeAlgorithm() = default;

  PDBAggregationPipeAlgorithm(const Handle<PDBSourcePageSetSpec> &source,
                              const Handle<PDBSinkPageSetSpec> &sink,
                              const Handle<Vector<PDBSourcePageSetSpec>> &secondarySources);

  bool setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) override;

  bool run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

  PDBPhysicalAlgorithmType getAlgorithmType() override;

 private:
  /**
   * Vector of pipelines that will run the pre-aggregation portion of this algorithm.
   * These will create the hashmaps which are sent to other worker nodes.
   * The pipelines will be built when you call setup on this object.
   * This must be null when sending this object.
   */
  std::shared_ptr<std::vector<PipelinePtr>> preAggPipelines = nullptr;

  /**
   * Vector of pipelines that will run the final aggregation portion of this algorithm.
   * These will receive the hashmaps from the pre-aggregation and finish the aggregation
   * locally. The pipelines will be built when you call setup on this object.
   * This must be null when sending this object.
   */
  std::shared_ptr<std::vector<PipelinePtr>> finalAggPipelines = nullptr;


  /**
   * Should we materialize the output of the final aggregation, or not?
   */
  bool shouldMaterialize = false;

};

}


#endif //PDB_PDBAGGREGATIONPIPEALGORITHM_H
