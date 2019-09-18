#pragma once

#include "ComputePlan.h"

namespace pdb {

class KeyComputePlan : public ComputePlan {

  std::vector<AtomicComputationPtr> getLeftPipelineComputations(AtomicComputationPtr &source);

public:

  explicit KeyComputePlan(LogicalPlanPtr myPlan);

  PipelinePtr buildHashPipeline(AtomicComputationPtr &source,
                                const PDBAbstractPageSetPtr &inputPageSet,
                                const PDBAnonymousPageSetPtr &outputPageSet,
                                std::map<ComputeInfoType, ComputeInfoPtr> &params);


};

}
