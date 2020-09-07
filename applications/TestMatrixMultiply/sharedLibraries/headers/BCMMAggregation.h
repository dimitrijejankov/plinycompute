#pragma once

#include "LambdaCreationFunctions.h"
#include "TRABlock.h"
#include "AggregateComp.h"
#include <DeepCopy.h>

namespace pdb {

namespace matrix {

class BCMMAggregation : public AggregateComp<BCMMAggregation, TRABlock, TRABlock, TRABlockMeta, TRABlockData> {
 public:

  ENABLE_DEEP_COPY

  // the key type must have == and size_t hash () defined
  static Lambda<TRABlockMeta> getKeyProjectionWithInputKey(Handle<TRABlockMeta> aggMe) {
    return makeLambdaFromSelf(aggMe);
  }

  // the value type must have + defined
  static Lambda<TRABlockData> getValueProjection(Handle<TRABlock> aggMe) {
    return makeLambdaFromMethod(aggMe, getValueRef);
  }

};

}
}