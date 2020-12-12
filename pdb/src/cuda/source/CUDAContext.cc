#include <CUDAContext.h>

namespace pdb{


    CUDADevice_t::CUDADevice_t(int device, size_t size) : device(device), size(size) {}

    void CUDADevice_t::init() {

    }

    size_t CUDADevice_t::registerThread() {
        auto tID = static_cast<ThreadID>(pthread_self());
        if (idxs.count(tID) == 0){
            return index++;
        } else {
            return idxs[tID];
        }
    }

    cublasHandle_t CUDADevice_t::getHandler() {
        auto idx = registerThread();
        if (idx >= 32){
            std::cerr << "CUDA ERROR: Run out of Handlers!\n";
            exit(-1);
        }
        return handles[idx];
    }

    cudaStream_t CUDADevice_t::getStream()  {
        auto idx = registerThread();
        if (idx >= 32){
            std::cerr << "CUDA ERROR: Run out of Handlers!\n";
            exit(-1);
        }
        return streams[idx];
    }


    CUDAContext::CUDAContext() {
        checkCudaErrors(cudaGetDeviceCount(&numDevices));
        devices.reserve(numDevices);
        for (int i =0; i < numDevices;i++){
            auto gpuDevice = std::make_unique<CUDADevice_t>(i, GPU_MEM_SIZE_RESERVERD);
            gpuDevice->mgr = std::make_unique<cudaMemMgr>();
            //TODO: choose which gpu to use
            for (int j = 0; j < gpuDevice->numStreams; j++){
                checkCudaErrors(cudaStreamCreate(&gpuDevice->streams[j]));
            }
            devices.push_back(std::move(gpuDevice));
        }
    }

    CUDAContext::~CUDAContext() {

    }

    template<Strategy s>
    GPUID CUDAContext::MapWorkerToGPU(){
        auto tID = static_cast<ThreadID>(pthread_self());
        if (tg.count(tID) != 0)
            return tg[tID];
        if (s == Strategy::NON_PREEMPTIVE){
            GPUID gID= tID%numDevices;
            tg.insert(std::make_pair(tID,gID));
            return gID;
        }
    }
}
