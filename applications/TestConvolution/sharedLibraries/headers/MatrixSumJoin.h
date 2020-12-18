#pragma once

#include <LambdaCreationFunctions.h>
#include <mkl_cblas.h>
#include "JoinComp.h"
#include "MatrixBlock.h"

namespace pdb::matrix {

class MatrixSumJoin : public JoinComp <MatrixSumJoin, MatrixBlock, MatrixBlock, MatrixBlock> {
 public:

  ENABLE_DEEP_COPY

  MatrixSumJoin() = default;

  // (in1.rowID == in2.rowID) && (in1.colID == in2.colID)
  static Lambda <bool> getKeySelection (Handle <MatrixBlockMeta> in1, Handle <MatrixBlockMeta> in2) {

    return (makeLambdaFromSelf(in1) == makeLambdaFromSelf (in2));
  }

  static Lambda <Handle<MatrixBlockMeta>> getKeyProjection(Handle <MatrixBlockMeta> in1, Handle <MatrixBlockMeta> in2) {
    return makeLambda (in1, in2, [] (Handle <MatrixBlockMeta> &in1, Handle <MatrixBlockMeta> &in2) {
      Handle<MatrixBlockMeta> out = makeObject<MatrixBlockMeta>(in1->rowID, in1->colID);
      return out;
    });
  }

  static Lambda <Handle<MatrixBlockData>> getValueProjection(Handle <MatrixBlockData> in1, Handle <MatrixBlockData> in2) {

    return makeLambda (in1, in2, [] (Handle <MatrixBlockData> &in1, Handle <MatrixBlockData> &in2) {

      // make an output
      Handle<MatrixBlockData> sum = makeObject<MatrixBlockData>(in1->numRows, in1->numCols);

      // get the ptrs
      float *sumData = sum->data->c_ptr();
      float *in1Data = in1->data->c_ptr();
      float *in2Data = in2->data->c_ptr();

      // do the sum
      int N = in1->numRows * in1->numCols;
      for(int32_t i = 0; i < N; ++i) {
        sumData[i] = in1Data[i] + in2Data[i];
      }

      // return the output
      return sum;
    });
  }
};


}