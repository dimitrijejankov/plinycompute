#ifndef PDB_CUDA_STATIC_STORAGE
#define PDB_CUDA_STATIC_STORAGE

#include <storage/PDBCUDAMemoryManager.h>
#include <PDBCUDAConfig.h>

/**
 * StaticStorage is for handling all the static space allocation. (input parameters)
 * The allocation unit is page
 */

//TODO: let invoke remember the input pages/output page
namespace pdb{

enum MemAllocateStatus{
    OLD,
    NEW
};

enum MemAllocatePolicy{
    DYNAMIC,
    STATIC
};

class PDBCUDAStaticStorage{

public:
    PDBCUDAStaticStorage() = default;

    size_t getObjectOffsetWithCPUPage(void* pageAddress, void* objectAddress);

    pair<void*, size_t> getCPUPageFromObjectAddress(void* objectAddress);

    //TODO: the policy of checkGPUPageTable should be decided based on the size of pageInfo
    // If size == Mgr.PageSize, use STATIC
    // If size < Mgr.PageSize, use DYNAMIC

    PDBCUDAPage* getGPUPageFromCPUPage(const pair<void*, size_t>& pageInfo, page_id_t* gpuPageID);

    inline bool IsCPUPageMovedToGPU(pair<void*, size_t> pageInfo);
    bool IsObjectOnGPU(void* objectAddress);

public:

    /**  H2DPageMap for mapping CPU bufferManager page info to GPU bufferManager page ids */
    /** Used for static allocation */
    std::map<pair<void *, size_t>, page_id_t> pageMap;

    /** one latch to protect the gpuPageTable access */
    std::mutex pageMapLatch;

    friend class PDBCUDAMemoryManager;
};
}

#endif