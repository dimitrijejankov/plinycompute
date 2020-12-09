
// All the global variables should be defined here.
#include <iostream>
#include <atomic>

#ifndef PDB_CUDA_GLOBAL_VARS
#define PDB_CUDA_GLOBAL_VARS

void* gpuMemoryManager = nullptr;
void* gpuStreamManager = nullptr;
void* gpuStaticStorage = nullptr;
void* gpuDynamicStorage = nullptr;

std::atomic<int> debugger{0};

#endif