#ifndef PDB_CUDA_CONFIG
#define PDB_CUDA_CONFIG

#include <cstdint>
#include <iostream>
#include <list>

namespace pdb{
    using frame_id_t  = int32_t;
    using ref_bit = bool;
    using frame_ref_info = std::pair<frame_id_t, ref_bit>;
    using buffer_iter = std::list<frame_ref_info>::iterator;
    static constexpr int32_t INVALID_PAGE_ID = -1;
    static constexpr int32_t CUDA_STREAM_NUM = 32;
    static constexpr int32_t CUDA_MEM_MAMAGER_PAGE_NUM = 9;
    static constexpr int32_t CPU_STORAGE_MANAGER_PAGE_NUM = 7;
    static constexpr size_t  CPU_STORAGE_MANAGER_PAGE_SIZE = 1024*1024*1024;
    static constexpr size_t  GPU_MEM_SIZE_RESERVERD = 1024*1024*1024*9;
    using GPUID = int;
    using ThreadID = long;
    using GPUPageID = std::size_t;
    using page_id_t = GPUPageID;
}
#endif