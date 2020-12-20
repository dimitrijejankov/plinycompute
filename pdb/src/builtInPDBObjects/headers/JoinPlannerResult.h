#pragma once

//  PRELOAD %JoinPlannerResult%

#include <Join3KeyPipeline.h>
#include <PDBVector.h>
#include "../../../../applications/TestConvolution/sharedLibraries/headers/MatrixBlockMeta.h"

using namespace pdb::matrix;

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
    Handle<pdb::Vector<pdb::Vector<int32_t>>> aggRecords;
    Handle<pdb::Vector<int32_t>> aggMapping;
    Handle<pdb::Map<MatrixBlockMeta, int32_t>> records0;
    Handle<pdb::Map<MatrixBlockMeta, int32_t>> records1;
    Handle<pdb::Map<MatrixBlockMeta, int32_t>> records2;
  };

}