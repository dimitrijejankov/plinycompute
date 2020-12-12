#ifndef PDB_CUDA_CPU_STORAGE_MANAGER
#define PDB_CUDA_CPU_STORAGE_MANAGER

#include <CUDAConfig.h>
#include <list>
#include <map>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "helper_cuda.h"
#include "CUDAUtility.h"
#include "CUDARamPointer.h"
#include "cudaMemMgr.h"

namespace pdb{

    enum Strategy{
        NON_PREEMPTIVE,
        PREEMPTIVE,
    };

    class CUDADevice_t{
    public:

        CUDADevice_t(int device, size_t size);

        void init();

        cublasHandle_t getHandler();

        cudaStream_t getStream();

        size_t registerThread();

    public:
        /** The device number. */
        GPUID device;

        /** The memory size to allocate for that device. If 0, the implementation chooses the size. */
        size_t size;

        /** The number of named streams associated with the device. The NULL stream is not counted. */
        int numStreams{32};

        /** The streams associated with the device. It can be NULL. The NULL stream is managed. */
        cudaStream_t streams[32];

        /** The handles associated with cublas calls. */
        cublasHandle_t handles[32];

        /** the memory manager on each devices */
        std::unique_ptr<cudaMemMgr> mgr;

    private:
        size_t index{0};
        std::map<ThreadID, size_t> idxs;
    };

    class CUDAContext{
    public:
        CUDAContext();
        ~CUDAContext();

        template<Strategy t>
        GPUID MapWorkerToGPU();

    private:
        std::vector<std::unique_ptr<CUDADevice_t> > devices;
        int numDevices;
        std::map<ThreadID, GPUID > tg;
    };
}
#endif