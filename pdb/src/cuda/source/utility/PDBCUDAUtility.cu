#include <assert.h>
#include "utility/PDBCUDAUtility.h"

void printCudaVersion() {
    std::cout << "CUDA Compiled version: " << __CUDACC_VER__ << std::endl;
    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;
    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}

void copyFromHostToDevice(void **targetDevice, void *sourceHost, size_t bytesNum) {
    checkCudaErrors(cudaMalloc((void **) targetDevice, bytesNum));
    checkCudaErrors(cudaMemcpy(*targetDevice, sourceHost, bytesNum, cudaMemcpyHostToDevice));
}

void copyFromHostToDeviceAsync(void *targetDevice, void *sourceHost, size_t bytesNum, cudaStream_t cs) {
    checkCudaErrors(cudaMemcpyAsync(targetDevice, sourceHost, bytesNum, cudaMemcpyHostToDevice, cs));
}

void copyFromDeviceToHost(void *targetHost, void *sourceDevice, size_t bytesNum) {
    checkCudaErrors(cudaMemcpy(targetHost, sourceDevice, bytesNum, cudaMemcpyDeviceToHost));
}

void copyFromDeviceToHostAsync(void *targetHost, void *sourceDevice, size_t bytesNum, cudaStream_t cs) {
    checkCudaErrors(cudaMemcpyAsync(targetHost, sourceDevice, bytesNum, cudaMemcpyDeviceToHost, cs));
}

void freeGPUMemory(void **memdata) {
    checkCudaErrors(cudaFree(*memdata));
}

void initGPUMemoryToZero(void **memdata, size_t bytesNum) {
    checkCudaErrors(cudaMalloc((void **) memdata, bytesNum));
    checkCudaErrors(cudaMemset(*memdata, 0, bytesNum));
}

int isDevicePointer(const void *ptr) {
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    cudaError_t err = cudaGetLastError();
    if (attributes.devicePointer != nullptr && err == 0) {
        return 1;
    } else {
        return 0;
    }
}



