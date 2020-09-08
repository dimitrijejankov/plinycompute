#pragma once

#include <memory>
#include <LogicalPlan.h>
#include <PipelineInterface.h>

namespace pdb {


struct PDBPhysicalAlgorithmState {

  virtual ~PDBPhysicalAlgorithmState() = default;

  /**
   * The logical plan
   */
  pdb::LogicalPlanPtr logicalPlan;

  std::shared_ptr<std::vector<PipelinePtr>> joinPipelines;

  /**
   * The logger of the algorithm
   */
  PDBLoggerPtr logger;

};

// make a shared ptr
using PDBPhysicalAlgorithmStatePtr = std::shared_ptr<PDBPhysicalAlgorithmState>;

}
