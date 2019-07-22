#pragma once

#include <LambdaCreationFunctions.h>
#include "JoinComp.h"
#include "MatrixBlock.h"

namespace pdb {

namespace matrix {

class MatrixMultiplyJoin : public JoinComp <MatrixMultiplyJoin, MatrixBlock, MatrixBlock, MatrixBlock> {
public:

  ENABLE_DEEP_COPY

  Lambda <bool> getSelection (Handle <MatrixBlock> in1, Handle <MatrixBlock> in2) {
    return (makeLambdaFromMethod (in1, getColID) == makeLambdaFromMethod (in2, getRowID));
  }

  Lambda <Handle <MatrixBlock>> getProjection (Handle <MatrixBlock> in1, Handle <MatrixBlock> in2) {
    return makeLambda (in1, in2, [] (Handle <MatrixBlock> &in1, Handle <MatrixBlock> &in2) {

      // get the sizes
      uint32_t I = in1->data.numRows;
      uint32_t J = in2->data.numCols;
      uint32_t K = in1->data.numCols;

      // make the output block
      Handle <MatrixBlock> out = makeObject<MatrixBlock>(in1->getRowID(), in2->getColID(), I, J);

      // get the ptrs
      float *outData = out->data.data->c_ptr();
      float *in1Data = in1->data.data->c_ptr();
      float *in2Data = in2->data.data->c_ptr();

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

}