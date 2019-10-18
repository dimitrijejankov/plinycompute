#pragma once

#define  __GPU__COMPUTATIONS

#include <LambdaCreationFunctions.h>
#include "JoinComp.h"
#include "MatrixBlock.h"
#include "PDBCUDAMatrixMultiple.h"
#include <mkl.h>

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
#ifdef __GPU__COMPUTATIONS
      uint32_t I = in1->data.numRows;
      uint32_t J = in2->data.numCols;
      // K and L should be equal
      uint32_t K = in1->data.numCols;
      uint32_t L = in2->data.numRows;
#else
      MKL_INT I = in1->data.numRows;
      MKL_INT J = in2->data.numCols;
      // K and L should be equal
      MKL_INT K = in1->data.numCols;
      MKL_INT L = in2->data.numRows;
#endif
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
      float * in1DataMKL = (float *) mkl_malloc(I * K * sizeof(float), 64);
      float * in2DataMKL = (float *) mkl_malloc(L * J * sizeof(float), 64);
      float * outDataMKL = (float *) mkl_malloc(I * J * sizeof(float), 64);

      cblas_scopy(I * K, in1DataCPU, 1, in1DataMKL,1);
      cblas_scopy(L * J, in2DataCPU, 1, in2DataMKL,1);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  I, J, K, 1, in1DataMKL, K, in2DataMKL, J, 0, outDataMKL, J);
      cblas_scopy(I * J, outDataMKL, 1, outDataCPU, 1);

#endif
      return out;
    });
  }
};


}

}
#define __GPU__COMPUTATIONS
