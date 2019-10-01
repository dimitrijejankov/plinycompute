#pragma once

#define  __GPU__COMPUTATIONS

#include <LambdaCreationFunctions.h>
#include "JoinComp.h"
#include "MatrixBlock.h"
#include "PDBCUDAMatrixMultiple.h"

namespace pdb {

namespace matrix {

class MatrixMultiplyJoin : public JoinComp <MatrixMultiplyJoin, MatrixBlock, MatrixBlock, MatrixBlock> {
public:

  ENABLE_DEEP_COPY

  MatrixMultiplyJoin() = default;

  static Lambda <bool> getKeySelection (Handle <MatrixBlockMeta> in1, Handle <MatrixBlockMeta> in2) {
    return (makeLambdaFromMember (in1, colID) == makeLambdaFromMember (in2, rowID));
  }

  static Lambda <Handle <MatrixBlock>> getProjection (Handle <MatrixBlock> in1, Handle <MatrixBlock> in2) {
    return makeLambda (in1, in2, [] (Handle <MatrixBlock> &in1, Handle <MatrixBlock> &in2) {
      // get the sizes
      uint32_t I = in1->data.numRows;
      uint32_t J = in2->data.numCols;

      // K and L should be equal
      uint32_t K = in1->data.numCols;
      uint32_t L = in2->data.numRows;

      // make the output block
      Handle <MatrixBlock> out = makeObject<MatrixBlock>(in1->getRowID(), in2->getColID(), I, J);

      // get the ptrs
      float *outDataCPU = out->data.data->c_ptr();
      float *in1DataCPU = in1->data.data->c_ptr();
      float *in2DataCPU = in2->data.data->c_ptr();

#ifdef __GPU__COMPUTATIONS
      float * outDataGPU;
      float * in1DataGPU;
      float * in2DataGPU;

      copyFromHostToDevice(&in1DataGPU, in1DataCPU, I, K);
      copyFromHostToDevice(&in2DataGPU, in2DataCPU, L, J);
      initGPUMemoryToZero(&outDataGPU, I, J);
      launchKernel(in1DataGPU, I, K, in2DataGPU, L, J, outDataGPU);
      copyFromDeviceToHost(outDataCPU, outDataGPU, I, J);
      freeGPUMemory(&in1DataGPU);
      freeGPUMemory(&in2DataGPU);
      freeGPUMemory(&outDataGPU);
#else
      //TODO replace this with mkl
      for (uint32_t i = 0; i < I; ++i) {
        for (uint32_t j = 0; j < J; ++j) {
          for (uint32_t k = 0; k < K; ++k) {
            outDataCPU[i * J + j] += in1DataCPU[i * K + k] * in2DataCPU[k * J + j];
          }
        }
      }
#endif
      return out;
    });
  }
};


}

}
#define __GPU__COMPUTATIONS
