#ifndef PDB_CUDA_CPU_STORAGE_MANAGER
#define PDB_CUDA_CPU_STORAGE_MANAGER

#include <PDBCUDAConfig.h>
#include <list>
#include <map>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "PDBCUDAUtility.h"
#include "PDBRamPointer.h"

namespace pdb{

    class PDBCUDACPUStorageManager{

    public:

        PDBCUDACPUStorageManager(int32_t PageNum = CPU_STORAGE_MANAGER_PAGE_NUM, size_t PageSize = CPU_STORAGE_MANAGER_PAGE_SIZE);

        ~PDBCUDACPUStorageManager();

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