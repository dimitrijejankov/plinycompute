#include "storage/PDBCUDAStaticStorage.h"

extern void* gpuMemoryManager;
namespace pdb{

    // TODO: we should have someway to "remember" input pages and unpin them.
    // TODO: principle: make sure when the computation is on, input/ouput pages are pinned. otherwise, pages should be unpinned.

    size_t PDBCUDAStaticStorage::getObjectOffsetWithCPUPage(void* pageAddress, void* objectAddress) {
        return (char *) objectAddress - (char *) pageAddress;
    }

    inline bool PDBCUDAStaticStorage::IsCPUPageMovedToGPU(pair<void*, size_t> pageInfo){
        return pageMap.find(pageInfo) != pageMap.end();
    }

    bool PDBCUDAStaticStorage::IsObjectOnGPU(void* objectAddress){
        auto pageInfo = getCPUPageFromObjectAddress(objectAddress);
        return pageMap.find(pageInfo) != pageMap.end();
    }

    pair<void*, size_t> PDBCUDAStaticStorage::getCPUPageFromObjectAddress(void* objectAddress) {
        // objectAddress must be a CPU RAM Pointer
        assert(isDevicePointer(objectAddress) == 0);
        pdb::PDBPagePtr whichPage = static_cast<PDBCUDAMemoryManager*>(gpuMemoryManager)->getCPUBufferManagerInterface()->getPageForObject(objectAddress);
        if (whichPage == nullptr) {
            std::cout << "getObjectCPUPage: cannot get page for this object!\n";
            exit(-1);
        }
        void *pageAddress = whichPage->getBytes();
        size_t pageBytes = whichPage->getSize();
        auto pageInfo = std::make_pair(pageAddress, pageBytes);
        return pageInfo;
    }


    PDBCUDAPage* PDBCUDAStaticStorage::getGPUPageFromCPUPage(const pair<void*, size_t>& pageInfo, page_id_t* gpuPageID){
        //TODO: We should change the mutex to a ReadWrite Lock

        std::lock_guard<std::mutex> guard(pageMapLatch);
        if (pageMap.find(pageInfo) != pageMap.end()){
            // return false means the GPU page is already created.
            PDBCUDAPage* cudaPage = static_cast<PDBCUDAMemoryManager*>(gpuMemoryManager)->FetchPageImplFromCPU(pageMap[pageInfo]);
            *gpuPageID = pageMap[pageInfo];
            return cudaPage;
        } else {
            // otherwise, grab a new page, insert to map and return pageID.
            page_id_t newPageID;
            PDBCUDAPage* cudaPage = static_cast<PDBCUDAMemoryManager*>(gpuMemoryManager)->NewPageImpl(&newPageID);
            *gpuPageID = newPageID;
            pageMap.insert(std::make_pair(pageInfo, newPageID));
            // return true means the GPU page is newly created.
            return cudaPage;
        }
    }

}