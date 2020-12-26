#pragma once

//  PRELOAD %JoinPlannerResult%

#include <Join3KeyPipeline.h>
#include <PDBVector.h>
#include "TRABlockMeta.h"


namespace pdb {

  class JoinPlannerResult : public Object {
  public:

    ~JoinPlannerResult() = default;

    JoinPlannerResult() = default;

    ENABLE_DEEP_COPY

    // this is the stuff we need to execute the query
    Handle<pdb::Vector<int32_t>> join_group_mapping;
    Handle<pdb::Vector<bool>> record_mapping;
    Handle<pdb::Vector<Join3KeyPipeline::joined_record>> joinedRecords;
    Handle<pdb::Vector< pdb::Handle<pdb::Vector<int32_t>> >> aggRecords;
    Handle<pdb::Vector<int32_t>> aggMapping;
    Handle<pdb::Map<TRABlockMeta, int32_t>> records0;
    Handle<pdb::Map<TRABlockMeta, int32_t>> records1;
    Handle<pdb::Map<TRABlockMeta, int32_t>> records2;
  };

}