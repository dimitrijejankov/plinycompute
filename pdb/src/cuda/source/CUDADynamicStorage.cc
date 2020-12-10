
#include "CUDADynamicStorage.h"
#include <CUDAMemoryManager.h>

namespace pdb{

    /*
    void* CUDADynamicStorage::memMalloc(size_t size){
        if (dynamicPages.size() == 0) {
            page_id_t newPageID;
            CUDAPage* newPage = CUDAMemoryManager::get()->NewPageImpl(&newPageID);
            bytesUsed = 0;
            pageSize = newPage->getPageSize();
            dynamicPages.push_back(newPageID);
            CUDAMemoryManager::get()->UnpinPageImpl(newPageID, false);
        }
        if (size > (pageSize - bytesUsed)) {
            page_id_t newPageID;
            CUDAMemoryManager::get()->NewPageImpl(&newPageID);
            bytesUsed = 0;
            dynamicPages.push_back(newPageID);
            CUDAMemoryManager::get()->UnpinPageImpl(newPageID, false);
        }
        size_t start = bytesUsed;
        bytesUsed += size;
        CUDAPage* currentPage = CUDAMemoryManager::get()->FetchEmptPageImpl(dynamicPages.back());
        return static_cast<void *>(currentPage->getBytes() + start) ;
    }

    void CUDADynamicStorage::memFree(void *ptr){
        //TODO: to be implemented
    }
    */

    /*
    pdb::RamPointerReference CUDADynamicStorage::keepMemAddress(void *gpuAddress, void *cpuAddress, size_t numBytes, size_t headerBytes){
        return addRamPointerCollection(gpuAddress, cpuAddress, numBytes,headerBytes);
    }
     */

    /*
    RamPointerReference CUDADynamicStorage::addRamPointerCollection(void *gpuAddress, void *cpuAddress, size_t numBytes = 0, size_t headerBytes = 0) {

        if (ramPointerCollection.count(gpuAddress) != 0) {
            ramPointerCollection[gpuAddress]->push_back_cpu_pointer(cpuAddress);

            //std::cout << " already exist RamPointerCollection size: " << ramPointerCollection.size() << std::endl;
            return std::make_shared<RamPointerBase>(ramPointerCollection[gpuAddress]);
        } else {
            RamPointerPtr ptr = std::make_shared<RamPointer>(gpuAddress, numBytes, headerBytes);
            ptr->push_back_cpu_pointer(cpuAddress);
            ramPointerCollection[gpuAddress] = ptr;
            //std::cout << " non exist RamPointerCollection size: " << ramPointerCollection.size() << std::endl;
            return std::make_shared<RamPointerBase>(ptr);
        }
    }
     */
}