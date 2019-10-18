#include <iostream>
#include <cstdio>
#include "PDBCUDAMatrixMultiple.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include <helper_cuda.h>

#define NUM_THREADS 8

__global__ void matrixMulGPU(float *in1data,
                             unsigned int in1NumRow,
                             unsigned int in1NumCol,
                             float *in2data,
                             unsigned int in2NumRow,
                             unsigned int in2NumCol,
                             float *outdata) {
  if (in1NumCol != in2NumRow) {
    return;
  }
  unsigned int I = in1NumRow;
  unsigned int J = in2NumCol;
  unsigned int K = in1NumCol;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < I && col < J) {
    for (int k = 0; k < K; ++k) {
      outdata[row * J + col] += in1data[row * K + k] * in2data[k * J + col];
    }
  }
}

void copyFromHostToDevice(float **targetDevice, float *sourceHost, unsigned int numRows, unsigned int numCols) {
  const unsigned int numElems = numRows * numCols;
  checkCudaErrors(cudaMalloc((void **) targetDevice, numElems * sizeof(float)));
  checkCudaErrors(cudaMemcpy(*targetDevice, sourceHost, numElems * sizeof(float), cudaMemcpyHostToDevice));
}

void copyFromDeviceToHost(float *targetHost, float *sourceDevice, unsigned int numRows, unsigned int numCols) {
  const unsigned int numElems = numRows * numCols;
  checkCudaErrors(cudaMemcpy(targetHost, sourceDevice, numElems * sizeof(float), cudaMemcpyDeviceToHost));
}

void launchKernel(float *in1data,
                  unsigned int in1NumRow,
                  unsigned int in1NumCol,
                  float *in2data,
                  unsigned int in2NumRow,
                  unsigned int in2NumCol,
                  float *outdataGPU) {
    cublasHandle_t handle;
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    dim3 threads_per_block(NUM_THREADS, NUM_THREADS, 1);
    dim3 number_of_blocks((in1NumRow / threads_per_block.x) + 1, (in2NumCol / threads_per_block.y) + 1, 1);
    checkCudaErrors(cublasCreate(&handle));
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in2NumCol, in1NumRow, in1NumCol, &alpha, in2data, in2NumCol, in1data, in1NumCol, &beta, outdataGPU, in2NumCol));
    //matrixMulGPU <<< number_of_blocks, threads_per_block >>> (in1data, in1NumRow, in1NumCol, in2data, in2NumRow, in2NumCol, outdataGPU);
}

void initGPUMemoryToZero(float **memdata, unsigned int numRows, unsigned int numCols) {
  const unsigned int numElems = numRows * numCols;
  checkCudaErrors(cudaMalloc((void **) memdata, numElems * sizeof(float)));
  checkCudaErrors(cudaMemset(*memdata, 0, numElems * sizeof(float)));
}

void printCudaVersion() {
  std::cout << "CUDA Compiled version: " << __CUDACC_VER__ << std::endl;
  int runtime_ver;
  cudaRuntimeGetVersion(&runtime_ver);
  std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;
  int driver_ver;
  cudaDriverGetVersion(&driver_ver);
  std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}

void freeGPUMemory(float ** memdata){
  checkCudaErrors(cudaFree(*memdata));
}
