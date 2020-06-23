#pragma once

#include "LambdaCreationFunctions.h"
#include "BPStrip.h"
#include "AggregateComp.h"
#include <DeepCopy.h>

namespace pdb {

namespace bp {

class BPMultiplyAggregation : public AggregateComp<BPMultiplyAggregation, BPStrip, BPStrip, BPStripMeta, BPStripData> {
 public:

  ENABLE_DEEP_COPY

  // the key type must have == and size_t hash () defined
  static Lambda<BPStripMeta> getKeyProjectionWithInputKey(Handle<BPStripMeta> aggMe) {
    return makeLambdaFromSelf(aggMe);
  }

  // the value type must have + defined
  static Lambda<BPStripData> getValueProjection(Handle<BPStrip> aggMe) {
    return makeLambdaFromMethod(aggMe, getValueRef);
  }

};

}
}