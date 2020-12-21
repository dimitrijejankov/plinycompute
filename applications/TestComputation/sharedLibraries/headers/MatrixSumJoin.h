#pragma once

#include <LambdaCreationFunctions.h>
//#include <mkl_cblas.h>
#include "JoinComp.h"
#include "TRABlock.h"

namespace pdb::matrix {

class MatrixSumJoin : public JoinComp <MatrixSumJoin, TRABlock, TRABlock, TRABlock> {
 public:

  ENABLE_DEEP_COPY

  MatrixSumJoin() = default;

  // (in1.rowID == in2.rowID) && (in1.colID == in2.colID)
  static Lambda <bool> getKeySelection (Handle <TRABlockMeta> in1, Handle <TRABlockMeta> in2) {

    return (makeLambdaFromSelf(in1) == makeLambdaFromSelf (in2));
  }

  static Lambda <Handle<TRABlockMeta>> getKeyProjection(Handle <TRABlockMeta> in1, Handle <TRABlockMeta> in2) {
    return makeLambda (in1, in2, [] (Handle <TRABlockMeta> &in1, Handle <TRABlockMeta> &in2) {
      Handle<TRABlockMeta> out = makeObject<TRABlockMeta>(in1->getIdx0(), in1->getIdx1());
      return out;
    });
  }

  static Lambda <Handle<TRABlockData>> getValueProjection(Handle <TRABlockData> in1, Handle <TRABlockData> in2) {

    return makeLambda (in1, in2, [] (Handle <TRABlockData> &in1, Handle <TRABlockData> &in2) {

      // make an output
      Handle<TRABlockData> sum = makeObject<TRABlockData>(in1->dim0, in1->dim1, 1);

      // get the ptrs
      float *sumData = sum->data->c_ptr();
      float *in1Data = in1->data->c_ptr();
      float *in2Data = in2->data->c_ptr();

      // do the sum
      int N = in1->dim0 * in1->dim1;
      for(int32_t i = 0; i < N; ++i) {
        sumData[i] = in1Data[i] + in2Data[i];
      }

      // return the output
      return sum;
    });
  }
};


}