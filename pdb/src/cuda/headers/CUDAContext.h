#ifndef PDB_CUDA_CPU_STORAGE_MANAGER
#define PDB_CUDA_CPU_STORAGE_MANAGER

#include <CUDAConfig.h>
#include <list>
#include <map>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "CUDAUtility.h"
#include "CUDARamPointer.h"

#define CNMEM_CHECK_TRUE(cond, error) do { \
    if( !(cond) ) { \
        CNMEM_DEBUG_ERROR("CNMEM_CHECK_TRUE evaluates to false\n"); \
        return error; \
    } \
} while(0)
namespace pdb{

    typedef enum
    {
        CNMEM_STATUS_SUCCESS = 0,
        CNMEM_STATUS_CUDA_ERROR,
        CNMEM_STATUS_INVALID_ARGUMENT,
        CNMEM_STATUS_NOT_INITIALIZED,
        CNMEM_STATUS_OUT_OF_MEMORY,
        CNMEM_STATUS_UNKNOWN_ERROR
    } cnmemStatus_t;

    typedef struct CUDADevice_t_{
        /** The device number. */
        int device;
        /** The size to allocate for that device. If 0, the implementation chooses the size. */
        size_t size;
        /** The number of named streams associated with the device. The NULL stream is not counted. */
        int numStreams;
        /** The streams associated with the device. It can be NULL. The NULL stream is managed. */
        cudaStream_t *streams;
        /** The size reserved for each streams. It can be 0. */
        size_t *streamSizes;
    } CUDADevice_t;

    cnmemStatus_t cnmemInit(int numDevices, const CUDADevice_t *devices, unsigned flags) {
        // Make sure we have at least one device declared.
        CNMEM_CHECK_TRUE(numDevices > 0, CNMEM_STATUS_INVALID_ARGUMENT);

        // Find the largest ID of the device.
        int maxDevice = 0;
        for( int i = 0 ; i < numDevices ; ++i ) {
            if( devices[i].device > maxDevice ) {
                maxDevice = devices[i].device;
            }
        }

        // Create the global context.
        cnmem::Context::create();
        cnmem::Context *ctx = cnmem::Context::get();

        // Allocate enough managers.
        CNMEM_CHECK_TRUE(maxDevice >= 0, CNMEM_STATUS_INVALID_ARGUMENT);
        std::vector<cnmem::Manager> &managers = ctx->getManagers();
        managers.resize(maxDevice+1);

        // Create a root manager for each device and create the children.
        int oldDevice;
        CNMEM_CHECK_CUDA(cudaGetDevice(&oldDevice));
        for( int i = 0 ; i < numDevices ; ++i ) {
            CNMEM_CHECK_CUDA(cudaSetDevice(devices[i].device));
            std::size_t size = devices[i].size;
            cudaDeviceProp props;
            CNMEM_CHECK_CUDA(cudaGetDeviceProperties(&props, devices[i].device));
            if( size == 0 ) {
                size = props.totalGlobalMem / 2;
            }
            CNMEM_CHECK_TRUE(
                    size > 0 && size < props.totalGlobalMem, CNMEM_STATUS_INVALID_ARGUMENT);

            cnmem::Manager &manager = ctx->getManager(devices[i].device);
            manager.setDevice(devices[i].device);
            manager.setFlags(flags);

            size = cnmem::ceilInt(size, CNMEM_GRANULARITY);
            CNMEM_CHECK(manager.reserve(size));

            for( int j = 0 ; j < devices[i].numStreams ; ++j ) {
                cnmem::Manager *child = new cnmem::Manager;
                child->setParent(&manager);
                child->setDevice(devices[i].device);
                child->setStream(devices[i].streams[j]);
                child->setFlags(flags & ~CNMEM_FLAGS_CANNOT_GROW);
                if( devices[i].streamSizes && devices[i].streamSizes[j] > 0 ) {
                    //https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#sequential-but-misaligned-access-pattern
                    //Align stream blocks so stream base addresses are alligned to CNMEM_GRANULARITY
                    devices[i].streamSizes[j] = cnmem::ceilInt(devices[i].streamSizes[j], CNMEM_GRANULARITY);
                    CNMEM_CHECK(child->reserve(devices[i].streamSizes[j]));
                }
                CNMEM_CHECK(manager.addChild(child));
            }
        }
        CNMEM_CHECK_CUDA(cudaSetDevice(oldDevice));
        return CNMEM_STATUS_SUCCESS;
    }


    class CUDAContext{

    public:

        CUDAContext(int32_t PageNum = CPU_STORAGE_MANAGER_PAGE_NUM, size_t PageSize = CPU_STORAGE_MANAGER_PAGE_SIZE);

        ~CUDAContext();

        void ReadPage(page_id_t page_id, char* page_data);

        page_id_t AllocatePage();

        void DeallocatePage(page_id_t page_id);

        void WritePage(page_id_t page_id, void *page_data);

        RamPointerReference handleInputObjectWithRamPointer(std::pair<void *, size_t> pageInfo, void *objectAddress, size_t size, cudaStream_t cs);

        RamPointerReference addRamPointerCollection(void *gpuaddress, void *cpuaddress, size_t numbytes = 0, size_t headerbytes = 0);

        void DeepCopyD2H(void* startLoc, size_t numBytes);

    private:

        std::atomic<page_id_t>  next_page_id_;

        size_t pageSize;

        size_t pageNum;

        std::list<void*> freeList;

        std::map<page_id_t, void*> storageMap;

        std::mutex latch;
    };
}
#endif