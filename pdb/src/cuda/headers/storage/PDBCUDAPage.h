#ifndef PDB_CUDA_PAGE
#define PDB_CUDA_PAGE

#include <cstdint>
#include <PDBCUDAConfig.h>
#include <cuda_runtime.h>

namespace pdb{
    /**
     * NONE: default value
     * STATIC_PAGE: usually the page for `in` parameter
     * DYNAMIC_PAGE: usually the page for `out` parameter, for serving RamPointer
     */
    enum PDBCUDAPageType{
        NONE,
        STATIC_PAGE,
        DYNAMIC_PAGE
    };

    class PDBCUDAPage{

    public:

        PDBCUDAPage() = default;

        ~PDBCUDAPage(){
            cudaFree(data);
        }

        inline void setBytes(char* loc) { data = loc;}

        inline char* getBytes() { return data;}

        inline size_t getPageSize(){ return page_size; }

        inline void setIsMoved(bool isMoved){ is_moved = isMoved; }

        /** @return whether the page has been moved to GPU */
        inline bool isMoved(){ return is_moved; }

        inline void setDirty(bool isDirty){ is_dirty = isDirty;}

        inline bool isDirty() { return is_dirty; }

        inline void setPageID(page_id_t id) { page_id = id;}

        /** @return the page id of this page */
        inline page_id_t GetPageId() { return page_id; }

        /** @return the pin count of this page */
        inline int GetPinCount() { return pin_count; }

        /** increase the pin count */
        inline void incrementPinCount() { pin_count++; }

        /** decrease the pin count */
        inline void decrementPinCount() { pin_count--;}

        inline void setPageType(PDBCUDAPageType type) { page_type = type;}

        inline void setPageSize(int size) { page_size = size;}

        inline void Reset(){ ResetMemory(); pin_count = 0; is_dirty = false; is_moved = false; page_id = INVALID_PAGE_ID;}

    private:

        inline void ResetMemory() {
            assert(data != nullptr);
            assert(page_size != 0);
            cudaMemset(data, 0, page_size);
        }

        PDBCUDAPageType page_type = NONE;
        page_id_t page_id = INVALID_PAGE_ID;

        char* data = nullptr;
        bool is_dirty = false;
        bool is_moved = false;
        int pin_count = 0;
        int page_size = 0;
    };
};


#endif