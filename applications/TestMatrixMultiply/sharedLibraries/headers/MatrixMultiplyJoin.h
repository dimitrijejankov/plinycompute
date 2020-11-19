#pragma once

#include <LambdaCreationFunctions.h>
#include "JoinComp.h"
#include "MatrixBlock.h"

namespace pdb::matrix {

class MatrixMultiplyJoin : public JoinComp <MatrixMultiplyJoin, MatrixBlock, MatrixBlock, MatrixBlock> {
 public:

  ENABLE_DEEP_COPY

  MatrixMultiplyJoin() = default;

  static Lambda <bool> getKeySelection (Handle <MatrixBlockMeta> in1, Handle <MatrixBlockMeta> in2) {
    return (makeLambdaFromMember (in1, colID) == makeLambdaFromMember (in2, rowID));
  }

  static Lambda <Handle<MatrixBlockMeta>> getKeyProjection(Handle <MatrixBlockMeta> in1, Handle <MatrixBlockMeta> in2) {
    return makeLambda (in1, in2, [] (Handle <MatrixBlockMeta> &in1, Handle <MatrixBlockMeta> &in2) {
      Handle<MatrixBlockMeta> out = makeObject<MatrixBlockMeta>(in1->rowID, in2->colID);
      return out;
    });
  }

  static Lambda <Handle<MatrixBlockData>> getValueProjection(Handle <MatrixBlockData> in1, Handle <MatrixBlockData> in2) {
    return makeLambda (in1, in2, [] (Handle <MatrixBlockData> &in1, Handle <MatrixBlockData> &in2) {

      // get the sizes
      uint32_t I = in1->numRows;
      uint32_t J = in2->numCols;
      uint32_t K = in1->numCols;

      // make an output
      Handle<MatrixBlockData> out = makeObject<MatrixBlockData>(I, J);
      
      // get the ptrs
      float *outData = out->data->c_ptr();
      float *in1Data = in1->data->c_ptr();
      float *in2Data = in2->data->c_ptr();

      // do the multiply
        //TODO replace this with mkl
        for (uint32_t i = 0; i < I; ++i) {
            for (uint32_t j = 0; j < J; ++j) {
                for (uint32_t k = 0; k < K; ++k) {
                    outData[i * J + j] += in1Data[i * K + k] * in2Data[k * J + j];
                }
            }
        }

      // return the output
      return out;
    });
  }
};


}