#ifndef PDB_CUDA_MATRIX_MULTIPLE
#define PDB_CUDA_MATRIX_MULTIPLE

#include <iostream>
#include <cstdio>
#include <boost/stacktrace.hpp>
#include <functional>
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"

#define cublasErrCheck(condition) \
    do { \
        cublasStatus_t status = condition; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cublas Error: error code : %d %s %d\n", status, __FILE__, __LINE__); \
            exit(-1); \
                        } \
            } while (0)

void copyFromHostToDevice(void **targetDevice, void *sourceHost, size_t bytesNum);

void copyFromDeviceToHost(void *targetHost, void *sourceDevice, size_t bytesNum);

void copyFromHostToDeviceAsync(void *targetDevice, void *sourceHost, size_t bytesNum, cudaStream_t cs);

void copyFromDeviceToHostAsync(void *targetHost, void *sourceDevice, size_t bytesNum, cudaStream_t cs);

void printCudaVersion();

void launchKernel(float *in1data, unsigned int in1NumRow, unsigned int in1NumCol, float *in2data, unsigned int in2NumRow,
             unsigned int in2NumCol, float *outdataGPU);

void initGPUMemoryToZero(void **memdata, size_t bytesNum);

void freeGPUMemory(void **memdata);

int isDevicePointer(const void *ptr);


#endif