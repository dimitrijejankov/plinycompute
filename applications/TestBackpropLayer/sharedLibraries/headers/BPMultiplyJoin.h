#pragma once

#include <LambdaCreationFunctions.h>
#include <mkl_cblas.h>
#include "JoinComp.h"
#include "BPStrip.h"

namespace pdb::bp {

class BPMultiplyJoin : public JoinComp <BPMultiplyJoin, BPStrip, BPStrip, BPStrip> {
 public:

  ENABLE_DEEP_COPY

  BPMultiplyJoin() = default;

  static Lambda <bool> getKeySelection (Handle <BPStripMeta> in1, Handle <BPStripMeta> in2) {
    return (makeLambdaFromMember (in1, batchID) == makeLambdaFromMember (in2, batchID));
  }

  static Lambda <Handle<BPStripMeta>> getKeyProjection(Handle <BPStripMeta> in1, Handle <BPStripMeta> in2) {
    return makeLambda (in1, in2, [] (Handle <BPStripMeta> &in1, Handle <BPStripMeta> &in2) {
      Handle<BPStripMeta> out = makeObject<BPStripMeta>(0);
      return out;
    });
  }

  static Lambda <Handle<BPStripData>> getValueProjection(Handle <BPStripData> in1, Handle <BPStripData> in2) {
    return makeLambda (in1, in2, [&] (Handle <BPStripData> &in1, Handle <BPStripData> &in2) {

      // get the sizes
      uint32_t I = in1->numCols;
      uint32_t J = in2->numCols;
      uint32_t K = in1->numRows;

      // make an output
      Handle<BPStripData> out = makeObject<BPStripData>(I, J);

      // get the ptrs
      float *outData = out->data->c_ptr();
      float *in1Data = in1->data->c_ptr();
      float *in2Data = in2->data->c_ptr();

      // do the multiply
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, I, J, K, 1.0f, in1Data, I, in2Data, J, 0.0f, outData, J);

      // return the output
      return out;
    });
  }
};


}