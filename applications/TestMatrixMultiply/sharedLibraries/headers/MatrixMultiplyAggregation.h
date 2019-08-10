#pragma once

#include "LambdaCreationFunctions.h"
#include "MatrixBlock.h"
#include "AggregateComp.h"
#include <DeepCopy.h>

namespace pdb {

namespace matrix {

class MatrixMultiplyAggregation : public AggregateComp<MatrixMultiplyAggregation, MatrixBlock, MatrixBlock, MatrixBlockMeta, MatrixBlockData> {
public:

  ENABLE_DEEP_COPY

  // the key type must have == and size_t hash () defined
  static Lambda<MatrixBlockMeta> getKeyProjectionWithInputKey(Handle<MatrixBlockMeta> aggMe) {
    return makeLambdaFromSelf(aggMe);
  }

  // the value type must have + defined
  static Lambda<MatrixBlockData> getValueProjection(Handle<MatrixBlock> aggMe) {
    return makeLambdaFromMethod(aggMe, getValueRef);
  }

};

}
}