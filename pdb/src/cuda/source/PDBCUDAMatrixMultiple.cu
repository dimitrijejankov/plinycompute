#include <iostream>
#include <cstdio>
#include "PDBCUDAMatrixMultiple.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

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
  gpuErrchk(cudaMalloc((void **) targetDevice, numElems * sizeof(float)));
  gpuErrchk(cudaMemcpy(*targetDevice, sourceHost, numElems * sizeof(float), cudaMemcpyHostToDevice));
}

void copyFromDeviceToHost(float *targetHost, float *sourceDevice, unsigned int numRows, unsigned int numCols) {
  const unsigned int numElems = numRows * numCols;
  gpuErrchk(cudaMemcpy(targetHost, sourceDevice, numElems * sizeof(float), cudaMemcpyDeviceToHost));
}

void launchKernel(float *in1data,
                  unsigned int in1NumRow,
                  unsigned int in1NumCol,
                  float *in2data,
                  unsigned int in2NumRow,
                  unsigned int in2NumCol,
                  float *outdataGPU) {
  dim3 threads_per_block(2, 2, 1);
  dim3 number_of_blocks((in1NumRow / threads_per_block.x) + 1, (in2NumCol / threads_per_block.y) + 1, 1);

  matrixMulGPU <<< number_of_blocks, threads_per_block >>> (in1data, in1NumRow, in1NumCol, in2data, in2NumRow, in2NumCol, outdataGPU);
}

void initGPUMemoryToZero(float **memdata, unsigned int numRows, unsigned int numCols) {
  const unsigned int numElems = numRows * numCols;
  gpuErrchk(cudaMalloc((void **) memdata, numElems * sizeof(float)));
  gpuErrchk(cudaMemset(*memdata, 0, numElems * sizeof(float)));
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

