#include <storage/PDBCUDACPUStorageManager.h>

namespace pdb{

    PDBCUDACPUStorageManager::PDBCUDACPUStorageManager(int32_t PageNum, size_t PageSize): pageNum(PageNum), pageSize(PageSize){
        for (size_t i = 0; i<PageNum;i++){
             void* page = malloc(PageSize);
             freeList.push_back(page);
        }
        next_page_id_ = 0;
    }

    PDBCUDACPUStorageManager::~PDBCUDACPUStorageManager() {
        for (const auto& toDelete: storageMap){
            free(toDelete.second);
        }
        for (const auto& toDelete: freeList){
            free(toDelete);
        }
    }

    void PDBCUDACPUStorageManager::ReadPage(page_id_t page_id, char* page_data){
        assert(isDevicePointer(page_data)==1);
        std::lock_guard<std::mutex> guard(latch);
        if (storageMap.find(page_id) == storageMap.end()){
            throw std::runtime_error("Cannot find the require page in CPU storage manager during readPage!");
        }
        void* page = storageMap[page_id];
        storageMap.erase(page_id);
        freeList.push_back(page);
        checkCudaErrors(cudaMemcpy(page_data, page, pageSize, cudaMemcpyHostToDevice));
    }

    void PDBCUDACPUStorageManager::WritePage(page_id_t page_id, void *page_data){
        assert(isDevicePointer(page_data)==1);
        std::lock_guard<std::mutex> guard(latch);
        if (storageMap.find(page_id) != storageMap.end()){
            throw std::runtime_error("Duplicate page in CPU storage manager during writePage!");
        }
        if (freeList.empty()){
            throw std::runtime_error("No available memory in CPU storage manager!");
        }
        void* page = freeList.front();
        freeList.pop_front();
        storageMap.insert(std::make_pair(page_id, page));
        checkCudaErrors(cudaMemcpy(page, page_data, pageSize, cudaMemcpyDeviceToHost));
    }

    page_id_t PDBCUDACPUStorageManager::AllocatePage() {
        return next_page_id_++;
    }

    void PDBCUDACPUStorageManager::DeepCopyD2H(void *startLoc, size_t numBytes) {
    }
}
