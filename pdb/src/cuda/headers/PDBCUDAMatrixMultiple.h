#ifndef PDB_CUDA_MATRIX_MULTIPLE
#define PDB_CUDA_MATRIX_MULTIPLE
#include <iostream>


extern void copyFromHostToDevice(float **targetDevice,
                                 float *sourceHost,
                                 unsigned int numRows,
                                 unsigned int numCols);

extern void copyFromDeviceToHost(float *targetHost,
                                 float *sourceDevice,
                                 unsigned int numRows,
                                 unsigned int numCols);

extern void printCudaVersion();

extern void launchKernel(float *in1data,
                         unsigned int in1NumRow,
                         unsigned int in1NumCol,
                         float *in2data,
                         unsigned int in2NumRow,
                         unsigned int in2NumCol,
                         float *outdataGPU);

extern void initGPUMemoryToZero(float **memdata,
                                unsigned int numRows,
                                unsigned int numCols);

extern void freeGPUMemory(float **memdata);

#endif