#pragma once

//  PRELOAD %JoinPlannerResult%

#include <Join3KeyPipeline.h>
#include <PDBVector.h>
#include "../../../applications/TestConvolution/sharedLibraries/headers/MatrixBlockMeta3D.h"

using namespace pdb::matrix_3d;

namespace pdb {

  class JoinPlannerResult : public Object {
  public:

    ~JoinPlannerResult() = default;

    JoinPlannerResult() = default;

    ENABLE_DEEP_COPY

    // this is the stuff we need to execute the query
    Handle<pdb::Vector<int32_t>> mapping;
    Handle<pdb::Vector<bool>> recordToNode;
    Handle<pdb::Vector<Join3KeyPipeline::joined_record>> joinedRecords;
    Handle<pdb::Map<MatrixBlockMeta3D, int32_t>> records;
  };

}