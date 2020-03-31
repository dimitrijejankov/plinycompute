#pragma once

#include "Computation.h"

namespace pdb {

class AggregateCompBase : public Computation {
 public:

  virtual ComputeSinkPtr getAggregationHashMapCombiner(size_t workerID) = 0;

  virtual PageProcessorPtr getAggregationKeyProcessor() = 0;

  virtual ComputeSinkPtr getKeyJoinAggSink(TupleSpec &consumeMe,
                                           TupleSpec &whichAttsToOpOn,
                                           TupleSpec &projection,
                                           const PDBPageHandle &aggKeyPage,
                                           std::map<ComputeInfoType, ComputeInfoPtr> &params,
                                           pdb::LogicalPlanPtr &plan) { return nullptr; }
};

}