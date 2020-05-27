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

class FFHiddenLayerJoin : public JoinComp <FFHiddenLayerJoin, FFMatrixBlock, FFMatrixBlock, FFMatrixBlock> {
 public:

  ENABLE_DEEP_COPY

  FFHiddenLayerJoin() = default;

  static Lambda <bool> getKeySelection (Handle <FFMatrixMeta> in1, Handle <FFMatrixMeta> in2) {
    return (makeLambdaFromMember (in1, colID) == makeLambdaFromMember (in2, rowID));
  }

  static Lambda <Handle<FFMatrixMeta>> getKeyProjection(Handle <FFMatrixMeta> in1, Handle <FFMatrixMeta> in2) {
    return makeLambda (in1, in2, [] (Handle <FFMatrixMeta> &in1, Handle <FFMatrixMeta> &in2) {
      Handle<FFMatrixMeta> out = makeObject<FFMatrixMeta>(in1->rowID, in2->colID);
      return out;
    });
  }

  static Lambda <Handle<FFMatrixData>> getValueProjection(Handle <FFMatrixData> in1, Handle <FFMatrixData> in2) {
    return makeLambda (in1, in2, [] (Handle <FFMatrixData> &in1, Handle <FFMatrixData> &in2) {

      // get the sizes
      uint32_t I = in1->numRows;
      uint32_t J = in2->numCols;
      uint32_t K = in1->numCols;

      // make an output
      Handle<FFMatrixData> out = makeObject<FFMatrixData>(I, J, in1->rowID, in2->colID);

      // get the ptrs
      float *outData = out->data->c_ptr();
      float *in1Data = in1->data->c_ptr();
      float *in2Data = in2->data->c_ptr();

      // do the relu on the activation this is left from the previous iteration
      for(int32_t i = 0; i < I * K; i++) {
        if(in1Data[i] < 0) { in1Data[i] = 0; }
      }

      // do the multiply
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, J, K, 1.0f, in1Data, K, in2Data, J, 0.0f, outData, J);

      // process the bias if necessary
      if(in2->bias != nullptr) {
        auto bias = in2->bias->c_ptr();
        for(uint32_t r = 0; r < I; r++) {
          for(uint32_t c = 0; c < J; c++) {

            // add the bias
            outData[r * J + c] += bias[c];
          }
        }
      }

      // return the output
      return out;
    });
  }
};

}

}