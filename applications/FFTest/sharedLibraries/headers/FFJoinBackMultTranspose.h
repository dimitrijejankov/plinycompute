#pragma once

#include <Object.h>
#include <Handle.h>
#include <LambdaCreationFunctions.h>
#include <JoinComp.h>
#include <mkl.h>
#include "FFMatrixMeta.h"
#include "FFMatrixData.h"
#include "FFMatrixBlock.h"

namespace pdb {

// the sub namespace
namespace ff {

class FFJoinBackMultTranspose : public JoinComp <FFJoinBackMultTranspose, FFMatrixBlock, FFMatrixBlock, FFMatrixBlock> {
 public:

  ENABLE_DEEP_COPY

  FFJoinBackMultTranspose() = default;

  // d_w2 = a_1 * trans(gradient_t)
  static Lambda <bool> getKeySelection (Handle <FFMatrixMeta> in1, Handle <FFMatrixMeta> in2) {
    return (makeLambdaFromMember (in1, colID) == makeLambdaFromMember (in2, colID));
  }

  static Lambda <Handle<FFMatrixMeta>> getKeyProjection(Handle <FFMatrixMeta> in1, Handle <FFMatrixMeta> in2) {
    return makeLambda (in1, in2, [] (Handle <FFMatrixMeta> &in1, Handle <FFMatrixMeta> &in2) {
      Handle<FFMatrixMeta> out = makeObject<FFMatrixMeta>(in1->rowID, in2->rowID);
      return out;
    });
  }

  static Lambda <Handle<FFMatrixData>> getValueProjection(Handle <FFMatrixData> in1, Handle <FFMatrixData> in2) {
    return makeLambda (in1, in2, [] (Handle <FFMatrixData> &in1, Handle <FFMatrixData> &in2) {

      // get the sizes
      uint32_t I = in1->numRows;
      uint32_t J = in2->numRows;
      uint32_t K = in1->numCols;
      uint32_t L = in2->numCols;

      // make an output
      Handle<FFMatrixData> out = makeObject<FFMatrixData>(I, J, in1->rowID, in2->rowID);

      // get the ptrs
      float *outData = out->data->c_ptr();
      float *in1Data_pre = in1->data->c_ptr();
      float *in1Data = (float*) mkl_malloc(sizeof(float) * in1->rowID * in1->colID, 64);
      float *in2Data = in2->data->c_ptr();

      // apply the relu
      for(int32_t i = 0; i < in1->rowID * in1->colID; ++i) {
        in1Data[i] = in1Data_pre[i] > 0 ? in1Data_pre[i] : 0;
      }

      // do the multiply
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, I, J, K, 1.0f, in1Data, K, in2Data, L, 0.0f, outData, J);

      // free the memory
      mkl_free(in1Data);

      // return the output
      return out;
    });
  }
};

}

}