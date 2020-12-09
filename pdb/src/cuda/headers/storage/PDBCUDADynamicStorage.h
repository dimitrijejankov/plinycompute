#ifndef PDB_CUDA_RAM_POINTER_MANAGER
#define PDB_CUDA_RAM_POINTER_MANAGER

#include <iostream>
#include <PDBCUDAConfig.h>
#include <mutex>
#include <vector>
#include <map>

/**
 * DynamicStorage is for handling all the dynamic space allocation. (e.g. RamPointer)
 * All the allocation/Free should use memMalloc() and memFree()
 * The allocation unit is byte
 */

namespace pdb{

//TODO: operations to memMalloc() should be implemented as thread safe.
//TODO: operations should be called from single stream.
class PDBCUDADynamicStorage{

public:

    PDBCUDADynamicStorage() = default;

    ~PDBCUDADynamicStorage() = default;

    // void* memMalloc(size_t size);

    // void memFree(void *ptr);

    //pdb::RamPointerReference keepMemAddress(void *gpuaddress, void *cpuaddress, size_t numbytes, size_t headerbytes);

    //RamPointerReference addRamPointerCollection(void *gpuaddress, void *cpuaddress, size_t numbytes = 0, size_t headerbytes = 0);

public:

    std::vector<page_id_t> dynamicPages;

    size_t bytesUsed = 0;

    size_t pageSize = 0;

    /**
     * This is a map between page_id and the RamPointer object. It keeps all the ramPointers we create on certain page
     */
    //std::map<page_id_t, pdb::RamPointerPtr> ramPointerCollection;

    friend class PDBCUDAMemoryManager;
};

}

#endif