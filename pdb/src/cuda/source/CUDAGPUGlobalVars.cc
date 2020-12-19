#ifndef PDB_CUDA_GLOBAL_VARS
#define PDB_CUDA_GLOBAL_VARS
// All the global variables should be defined here.
#include <atomic>

//void* gpuMemoryManager = nullptr;
//void* gpuStreamManager = nullptr;
//void* gpuStaticStorage = nullptr;
//void* gpuDynamicStorage = nullptr;
void* gpuContext = nullptr;
std::atomic<int> debugger{0};
#endif