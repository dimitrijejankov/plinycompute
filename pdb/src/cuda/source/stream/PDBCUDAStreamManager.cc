#include <assert.h>
#include "stream/PDBCUDAStreamManager.h"

namespace pdb {

    PDBCUDAStreamManager::PDBCUDAStreamManager() {
        streamNum = CUDA_STREAM_NUM;
        streams = new cudaStream_t[streamNum];
        handles = new cublasHandle_t[streamNum];
        for (uint32_t i = 0; i < streamNum; i++) {
            cudaStreamCreate(&streams[i]);
            cublasCreate(&handles[i]);
            cublasSetStream(handles[i], streams[i]);
        }
    }

    PDBCUDAStreamManager::~PDBCUDAStreamManager() {
        for (int i = 0; i < streamNum; i++) {
            cudaStreamDestroy(streams[i]);
            cublasDestroy(handles[i]);
        }
    }

    PDBCUDAStreamUtils PDBCUDAStreamManager::bindCPUThreadToStream() {
        long threadID = (long) pthread_self();
        if (bindMap.count(threadID) != 0) {
            //std::cout << "thread ID: " << threadID << " find in map! stream: " << streams[threadStreamMap[threadID]] << std::endl;
            return std::make_pair(streams[bindMap[threadID]], handles[bindMap[threadID]]);
        } else {
            uint64_t counter = bindMap.size();
            bindMap.insert(std::make_pair(threadID, counter));
            //std::cout << "thread ID: " << threadID << " not find in map stream: " << streams[counter] << std::endl;
            return std::make_pair(streams[counter], handles[counter]);
        }
    }
}
