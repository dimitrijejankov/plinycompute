#pragma once

#include "LambdaCreationFunctions.h"
#include "MatrixBlock.h"
#include "AggregateComp.h"
#include <DeepCopy.h>

namespace pdb {

namespace matrix {

class MatrixMultiplyAggregation : public AggregateComp<MatrixBlock, MatrixBlock, MatrixBlockMeta, MatrixBlockData> {
public:

  ENABLE_DEEP_COPY

  // the key type must have == and size_t hash () defined
  Lambda<MatrixBlockMeta> getKeyProjection(Handle<MatrixBlock> aggMe) override {
    return makeLambdaFromMethod(aggMe, getKey);
  }

  // the value type must have + defined
  Lambda<MatrixBlockData> getValueProjection(Handle<MatrixBlock> aggMe) override {
    return makeLambdaFromMethod(aggMe, getValue);
  }

};

}
}