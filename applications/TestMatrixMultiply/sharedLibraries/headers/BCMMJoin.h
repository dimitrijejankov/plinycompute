#pragma once

#include <LambdaCreationFunctions.h>
#include <mkl_cblas.h>
#include "JoinComp.h"
#include "TRABlock.h"

namespace pdb::matrix {

    //Todo:: Change this to local join
class BCMMJoin : public JoinComp <BCMMJoin, TRABlock, TRABlock, TRABlock> {
 public:

  ENABLE_DEEP_COPY

  BCMMJoin() = default;

  static Lambda <bool> getKeySelection (Handle <TRABlockMeta> in1, Handle <TRABlockMeta> in2) {
    return (makeLambdaFromMethod (in1, getIdx1) == makeLambdaFromMethod (in2, getIdx0));
  }

  static Lambda <Handle<TRABlockMeta>> getKeyProjection(Handle <TRABlockMeta> in1, Handle <TRABlockMeta> in2) {
    return makeLambda (in1, in2, [] (Handle <TRABlockMeta> &in1, Handle <TRABlockMeta> &in2) {
      Handle<TRABlockMeta> out = makeObject<TRABlockMeta>(in1->getIdx0(), in2->getIdx1(), 0);
      return out;
    });
  }

  static Lambda <Handle<TRABlockData>> getValueProjection(Handle <TRABlockData> in1, Handle <TRABlockData> in2) {
    return makeLambda (in1, in2, [] (Handle <TRABlockData> &in1, Handle <TRABlockData> &in2) {

      // get the sizes
      uint32_t I = in1->dim0;
      uint32_t J = in2->dim1;
      uint32_t K = in1->dim1;

      // make an output
      Handle<TRABlockData> out = makeObject<TRABlockData>(I, J, 0);
      
      // get the ptrs
      float *outData = out->data->c_ptr();
      float *in1Data = in1->data->c_ptr();
      float *in2Data = in2->data->c_ptr();

      // do the multiply
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, J, K, 1.0f, in1Data, K, in2Data, J, 0.0f, outData, J);

      // return the output
      return out;
    });
  }
};


}