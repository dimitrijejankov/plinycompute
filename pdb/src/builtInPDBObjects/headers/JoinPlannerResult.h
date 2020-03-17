#pragma once

//  PRELOAD %JoinPlannerResult%

#include <EightWayJoinPipeline.h>

namespace pdb {

  class JoinPlannerResult : public Object {
  public:

    ~JoinPlannerResult() = default;

    JoinPlannerResult() = default;

    ENABLE_DEEP_COPY

    // this is the stuff we need to execute the query
    Handle<pdb::Vector<int32_t>> mapping;
    Handle<pdb::Vector<EightWayJoinPipeline::joined_record>> joinedRecords;
    Handle<pdb::Vector<EightWayJoinPipeline::key>> records;
  };

}