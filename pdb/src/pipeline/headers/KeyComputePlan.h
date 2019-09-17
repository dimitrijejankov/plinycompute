#pragma once

#include "ComputePlan.h"

namespace pdb {

class KeyComputePlan : public ComputePlan {

  std::vector<AtomicComputationPtr> getLeftPipelineComputations(AtomicComputationPtr &source,
                                                                std::shared_ptr<LogicalPlan> &logicalPlan);

public:
//
//  PipelinePtr buildLeftPipeline(AtomicComputationPtr &source,
//                                const PDBAbstractPageSetPtr &inputPageSet,
//                                const PDBAnonymousPageSetPtr &outputPageSet,
//                                std::map<ComputeInfoType, ComputeInfoPtr> &params,
//                                std::shared_ptr<LogicalPlan> &myPlan);


};

}
