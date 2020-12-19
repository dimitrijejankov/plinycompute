#ifndef PDB_CUDA_MEM_MANAGER
#define PDB_CUDA_MEM_MANAGER

#include <map>
#include <list>
#include <set>
#include <assert.h>
#include <PDBBufferManagerInterface.h>
#include "CUDAUtility.h"
#include "CUDARamPointer.h"
#include "CUDAPage.h"
#include "CUDAConfig.h"
#include "CUDAReplacer.h"
#include "CUDAContext.h"
#include "CUDAConfig.h"

namespace pdb {

    class CUDAMemMgr;
    using PDBCUDAMemoryManagerPtr = std::shared_ptr<CUDAMemMgr>;
    class CUDAMemMgr {
    public:

        CUDAMemMgr()=default;

        /**
         * @param buffer - CPU buffer
         */
        CUDAMemMgr(pdb::PDBBufferManagerInterfacePtr buffer, GPUID deviceID) {
            bufferManager = buffer;
            poolSize = CUDA_MEM_MAMAGER_PAGE_NUM;
            pageSize = buffer->getMaxPageSize();
            pages = new CUDAPage[poolSize];
            replacer = new ClockReplacer(poolSize);

            checkCudaErrors(cudaSetDevice(deviceID));
            for (size_t i = 0; i < poolSize; i++) {
                void *cudaPointer = nullptr;
                checkCudaErrors(cudaMalloc((void **) &cudaPointer, pageSize));
                pages[i].setBytes(static_cast<char*>(cudaPointer));
                pages[i].setPageSize(pageSize);
                freeList.emplace_back(static_cast<int32_t>(i));
            }
            //cpu_storage_manager = new CUDAContext();
        }

        ~CUDAMemMgr() {
            delete[] pages;
            delete replacer;
            delete cpu_storage_manager;
        }

        void FindFramePlacement(frame_id_t& frame){
            // Try to find the available frame from free_list, if not, try to get from replacer
            if (!freeList.empty()) {
                frame = freeList.front();
                freeList.pop_front();
                return;
            }
            // find the available frame from replacer
            if (replacer->Size() > 0) {
                replacer->Victim(&frame);
                auto res = std::find_if(pageTable.begin(), pageTable.end(),
                                        [&](const std::pair<page_id_t, frame_id_t>& it) {
                                            return it.second == frame;
                                        });
                // if page is dirty, write it back to disk
                page_id_t toWriteBack = res->first;
                if (pages[frame].isDirty()) {
                    FlushPageImpl(toWriteBack);
                }
                pageTable.erase(res);
            } else {
                std::cout << "No available memory in FindFramePlacement()!\n";
                exit(-1);
            }
        }

        bool UnpinPageImpl(page_id_t page_id, bool is_dirty) {

            std::lock_guard<std::mutex> guard(latch);

            auto iter = pageTable.find(page_id);
            if (iter == pageTable.end()){
                return false;
            }
            auto& thisPage = pages[iter->second];
            auto pincount = thisPage.GetPinCount();
            if (pincount<=0){
                return false;
            }
            thisPage.setDirty(is_dirty);
            thisPage.decrementPinCount();
            if (thisPage.GetPinCount()==0){
                replacer->Unpin(iter->second);
            }
            return true;
        }

        bool IsAllPagesPinned(){
            return pageTable.size() == poolSize && replacer->Size() == 0;
        }

        /**
         * Similar to @FetchPageImpl, except read from CPU RAM.
         * @param page_id
         * @return
         */
        CUDAPage* FetchPageImpl(page_id_t page_id){
            std::lock_guard<std::mutex> guard(latch);

            std::cout << "print Page Table\n";
            for (const auto& p: pageTable){
                std::cout << "page id: "<< p.first << "frame id: " << p.second << std::endl;
            }

            auto iter = pageTable.find(page_id);
            if (iter != pageTable.end()){
                replacer->Pin(iter->second);
                pages[iter->second].incrementPinCount();
                return &pages[iter->second];
            }
            if (IsAllPagesPinned()){
                throw std::runtime_error("Not enough GPU Memory! All cuda pages have been pinned!\n");
            }
            // if cannot find the page from page_table, then try to get the page from free_list or replacer.
            frame_id_t replacement;
            FindFramePlacement(replacement);
            //update the page table
            pageTable[page_id] = replacement;
            pages[replacement].Reset();
            pages[replacement].incrementPinCount();
            pages[replacement].setPageID(page_id);
            replacer->Pin(replacement);
            //TODO: change cpu_storage_manager
            cpu_storage_manager->ReadPage(page_id, pages[replacement].getBytes());
            return &pages[replacement];
        }


        CUDAPage* FetchEmptyPageImpl(page_id_t page_id){
            std::lock_guard<std::mutex> guard(latch);

            auto iter = pageTable.find(page_id);
            if (iter != pageTable.end()){
                replacer->Pin(iter->second);
                pages[iter->second].incrementPinCount();
                return &pages[iter->second];
            }
            if (IsAllPagesPinned()){
                return nullptr;
            }
            // if cannot find the page from page_table, then try to get the page from free_list or replacer.
            frame_id_t replacement;
            FindFramePlacement(replacement);
            //update the page table
            pageTable[page_id] = replacement;
            pages[replacement].Reset();
            pages[replacement].incrementPinCount();
            pages[replacement].setPageID(page_id);
            replacer->Pin(replacement);
            // TODO: change cpu_storage_manager
            // cpu_storage_manager->ReadPage(page_id, pages[replacement].getBytes());
            if (pages[replacement].getBytes() == nullptr){
                exit(-1);
            }
            return &pages[replacement];
        }

        CUDAPage* NewPageImpl(page_id_t *page_id) {
            std::lock_guard<std::mutex> guard(latch);
            // Note: new page should not read from CPU storage!
            *page_id = cpu_storage_manager->AllocatePage();
            if (IsAllPagesPinned()){
                return nullptr;
            }
            // find a available frame to place the page
            frame_id_t placement;
            FindFramePlacement(placement);
            pageTable[*page_id] = placement;
            pages[placement].Reset();
            pages[placement].incrementPinCount();
            pages[placement].setPageID(*page_id);
            replacer->Pin(placement);
            return &pages[placement];
        }


        void CreateNewPage(page_id_t *page_id){
            *page_id = cpu_storage_manager->AllocatePage();
        }

        bool DeletePageImpl(page_id_t page_id) {

            std::lock_guard<std::mutex> guard(latch);
            //TODO: change cpu_storage_manager
            cpu_storage_manager->DeallocatePage(page_id);
            auto iter = pageTable.find(page_id);
            if (iter==pageTable.end()){
                return true;
            } else {
                auto& thisPage = pages[iter->second];
                auto pincount = thisPage.GetPinCount();
                if (pincount==0){
                    freeList.push_back(iter->second);
                    pageTable.erase(iter);
                    thisPage.Reset();
                    return true;
                } else {
                    return false;
                }
            }
        }

        bool FlushPageImpl(page_id_t page_id) {
            // if page_id is not valid
            assert(page_id != INVALID_PAGE_ID);
            // if page_id cannot be find from page_table
            auto iter = pageTable.find(page_id);
            if (iter == pageTable.end()){
                return false;
            }
            cpu_storage_manager->WritePage(page_id, pages[iter->second].getBytes());
            // Make sure you call DiskManager::WritePage!
            return true;
        }

        PDBBufferManagerInterfacePtr getCPUBufferManagerInterface(){
            return bufferManager;
        }


    public:

        /**
         * the buffer manager to help maintain gpu page table
         */
        PDBBufferManagerInterfacePtr bufferManager;

        /** Page table for keeping track of buffer pool pages. */
        std::unordered_map<page_id_t, frame_id_t> pageTable;

        /** array of all the pages */
        CUDAPage* pages;

        /** the size of the pool  */
        size_t poolSize;

        /** the size of the page in gpu Buffer Manager  */
        size_t pageSize;

        /** here is the replacer for */
        ClockReplacer *replacer;

        /** Frames for holding the free memory */
        std::list<frame_id_t> freeList;

        CUDAContext* cpu_storage_manager;

        std::mutex latch;
    };

}

#endif