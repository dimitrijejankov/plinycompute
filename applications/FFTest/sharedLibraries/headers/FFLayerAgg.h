#pragma once

#include <Object.h>
#include <Handle.h>
#include <LambdaCreationFunctions.h>
#include "FFMatrixMeta.h"
#include "FFMatrixData.h"
#include "FFMatrixBlock.h"
#include "AggregateComp.h"

namespace pdb {

// the sub namespace
namespace ff {

class FFLayerAgg : public AggregateComp<FFLayerAgg, FFMatrixBlock, FFMatrixBlock, FFMatrixMeta, FFMatrixData> {
 public:

  ENABLE_DEEP_COPY

  // the key type must have == and size_t hash () defined
  static Lambda<FFMatrixMeta> getKeyProjectionWithInputKey(Handle<FFMatrixMeta> aggMe) {
    return makeLambdaFromSelf(aggMe);
  }

  // the value type must have + defined
  static Lambda<FFMatrixData> getValueProjection(Handle<FFMatrixBlock> aggMe) {
    return makeLambdaFromMethod(aggMe, getValueRef);
  }

};

}
}