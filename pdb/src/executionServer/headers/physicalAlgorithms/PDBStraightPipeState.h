#pragma once

#include "PDBPhysicalAlgorithmState.h"
#include "Pipeline.h"

namespace pdb {

struct PDBStraightPipeState : public PDBPhysicalAlgorithmState {

  /**
   * Vector of pipelines that will run this algorithm. The pipelines will be built when you call setup on this object.
   * This must be null when sending this object.
   */
  std::shared_ptr<std::vector<PipelinePtr>> myPipelines = nullptr;

};

}
