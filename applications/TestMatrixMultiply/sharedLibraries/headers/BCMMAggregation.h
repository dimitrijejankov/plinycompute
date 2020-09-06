#pragma once

#include "LambdaCreationFunctions.h"
#include "TensorBlock.h"
#include "AggregateComp.h"
#include <DeepCopy.h>

namespace pdb {

namespace matrix {

class BCMMAggregation : public AggregateComp<BCMMAggregation, TensorBlock, TensorBlock, TensorBlockMeta, TensorBlockData> {
 public:

  ENABLE_DEEP_COPY

  // the key type must have == and size_t hash () defined
  static Lambda<TensorBlockMeta> getKeyProjectionWithInputKey(Handle<TensorBlockMeta> aggMe) {
    return makeLambdaFromSelf(aggMe);
  }

  // the value type must have + defined
  static Lambda<TensorBlockData> getValueProjection(Handle<TensorBlock> aggMe) {
    return makeLambdaFromMethod(aggMe, getValueRef);
  }

};

}
}