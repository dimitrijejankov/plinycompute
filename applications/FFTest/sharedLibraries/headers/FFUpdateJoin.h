#pragma once

#include <Object.h>
#include <Handle.h>
#include <LambdaCreationFunctions.h>
#include <JoinComp.h>
#include <mkl>
#include "FFMatrixMeta.h"
#include "FFMatrixData.h"
#include "FFMatrixBlock.h"

namespace pdb {

// the sub namespace
namespace ff {

class FFUpdateJoin : public JoinComp <FFUpdateJoin, FFMatrixBlock, FFMatrixBlock, FFMatrixBlock> {
 public:

  ENABLE_DEEP_COPY

  FFUpdateJoin() = default;

  static Lambda <bool> getKeySelection (Handle <FFMatrixMeta> in1, Handle <FFMatrixMeta> in2) {
    return makeLambdaFromSelf(in1) == makeLambdaFromSelf(in2);
  }

  static Lambda <Handle<FFMatrixMeta>> getKeyProjection(Handle <FFMatrixMeta> in1, Handle <FFMatrixMeta> in2) {
    return makeLambda (in1, in2, [] (Handle <FFMatrixMeta> &in1, Handle <FFMatrixMeta> &in2) {
      Handle<FFMatrixMeta> out = makeObject<FFMatrixMeta>(in1->rowID, in1->colID);
      return out;
    });
  }

  static Lambda <Handle<FFMatrixData>> getValueProjection(Handle <FFMatrixData> in1, Handle <FFMatrixData> in2) {
    return makeLambda (in1, in2, [] (Handle <FFMatrixData> &in1, Handle <FFMatrixData> &in2) {

      // get the sizes
      uint32_t I = in1->numRows;
      uint32_t K = in1->numCols;

      // make an output
      Handle<FFMatrixData> out = makeObject<FFMatrixData>(I, K, in1->rowID, in1->colID);

      auto data = out->data->c_ptr();
      auto lhs = in1->data->c_ptr();
      auto rhs = in2->data->c_ptr();

      // do the multiply
      for(int32_t i = 0; i < I * K; i++) {
        data[i] = lhs[i] + rhs[i];
      }

      if(in1->bias != nullptr && in2->bias != nullptr) {
        out->bias = pdb::makeObject<Vector<float>>(in1->bias->size(), in1->bias->size());
        float *o = out->bias->c_ptr();
        float *b1 = in1->bias->c_ptr();
        float *b2 = in2->bias->c_ptr();

        // sum update the bias
        for(int i = 0; i < in1->bias->size(); i++) {
          o[i] = b1[i] + b2[i];
        }
      }

      // return the output
      return out;
    });
  }
};

}

}