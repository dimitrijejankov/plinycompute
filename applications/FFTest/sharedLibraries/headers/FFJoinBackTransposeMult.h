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

class FFJoinBackTransposeMult : public JoinComp <FFJoinBackTransposeMult, FFMatrixBlock, FFMatrixBlock, FFMatrixBlock> {
 public:

  ENABLE_DEEP_COPY

  FFJoinBackTransposeMult() = default;
  explicit FFJoinBackTransposeMult(uint32_t numRows) : numRows(numRows) {};

  uint32_t numRows;

  static Lambda <bool> getKeySelection (Handle <FFMatrixMeta> in1, Handle <FFMatrixMeta> in2) {
    return (makeLambdaFromMember (in1, rowID) == makeLambdaFromMember (in2, rowID));
  }

  static Lambda <Handle<FFMatrixMeta>> getKeyProjection(Handle <FFMatrixMeta> in1, Handle <FFMatrixMeta> in2) {
    return makeLambda (in1, in2, [] (Handle <FFMatrixMeta> &in1, Handle <FFMatrixMeta> &in2) {
      Handle<FFMatrixMeta> out = makeObject<FFMatrixMeta>(in1->colID, in2->colID);
      return out;
    });
  }

  Lambda <Handle<FFMatrixData>> getValueProjection(Handle <FFMatrixData> in1, Handle <FFMatrixData> in2) {
    return makeLambda (in1, in2, [&] (Handle <FFMatrixData> &in1, Handle <FFMatrixData> &in2) {

      // get the sizes
      uint32_t I = in1->numCols;
      uint32_t J = in2->numCols;
      uint32_t K = in1->numRows;

      // make an output
      Handle<FFMatrixData> out = makeObject<FFMatrixData>(I, J, in1->colID, in2->colID);

      // get the ptrs
      float *outData = out->data->c_ptr();
      float *in1Data = in1->data->c_ptr();
      float *in2Data = in2->data->c_ptr();

      // do I need to do some summing here
      if(in1->colID == (numRows - 1)) {

        // make the bias
        out->bias = pdb::makeObject<Vector<float>>(J, J);
        auto bias = out->bias->c_ptr();
        for(auto c = 0; c < J; c++) {
          bias[c] = 0.0f;
        }

        // sum up the value
        for(auto r = 0; r < in2->numRows; r++) {
          for(auto c = 0; c < in2->numCols; c++) {
            bias[c] += in2Data[r * in2->numCols + c];
          }
        }
      }

      // do the multiply
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, I, J, K, 1.0f, in1Data, K, in2Data, J, 0.0f, outData, J);

      // return the output
      return out;
    });
  }
};

}

}