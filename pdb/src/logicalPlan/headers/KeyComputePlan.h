#pragma once

#include "ComputePlan.h"

namespace pdb {

class KeyComputePlan : public ComputePlan {

  std::vector<AtomicComputationPtr> getLeftPipelineComputations(AtomicComputationPtr &source);

public:

  explicit KeyComputePlan(LogicalPlanPtr myPlan);

  pdb::ComputeSourcePtr getKeySource(AtomicComputationPtr &sourceAtomicComputation,
                                     const PDBAbstractPageSetPtr &inputPageSet,
                                     std::map<ComputeInfoType, ComputeInfoPtr> &params);

  // returns the compute sink
  ComputeSinkPtr getJoinAggSink(AtomicComputationPtr &targetAtomicComp,
                                const std::string &targetComputationName,
                                std::map<ComputeInfoType, ComputeInfoPtr> &params);

  PipelinePtr buildHashPipeline(const std::string &sourceTupleSet,
                                const PDBAbstractPageSetPtr &inputPageSet,
                                const PDBAnonymousPageSetPtr &outputPageSet,
                                std::map<ComputeInfoType, ComputeInfoPtr> &params);


  PipelinePtr buildJoinAggPipeline(const std::string& sourceTupleSetName,
                                   const std::string& targetTupleSetName,
                                   const PDBAbstractPageSetPtr &inputPageSet,
                                   const PDBAnonymousPageSetPtr &outputPageSet,
                                   std::map<ComputeInfoType, ComputeInfoPtr> &params,
                                   size_t numNodes,
                                   size_t numProcessingThreads,
                                   uint64_t chunkSize,
                                   uint64_t workerID);
};

}
