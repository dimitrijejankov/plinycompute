#ifndef PDB_CUDA_VA_INVOKER
#define PDB_CUDA_VA_INVOKER

#include <iostream>
#include <Handle.h>
#include "PDBVector.h"
#include <functional>
#include <numeric>
#include "CUDAUtility.h"
#include "CUDAOpType.h"
#include "CUDAOpInvoker.h"
#include "CUDAStaticStorage.h"
#include "CUDAStreamManager.h"

namespace pdb {

    // simply support vector-add operation and vector-add kernel for GPU
    class CUDAVectorAddInvoker : public PDBCUDAInvoker {

    public:

        CUDAVectorAddInvoker();

        ~CUDAVectorAddInvoker();

        bool invoke();

        void kernel(float* in1data, float* in2data, size_t N);

        void setInput(float* input, const std::vector<size_t>& inputDim);

        void setOutput(float* output, const std::vector<size_t>& outputDim);

        //TODO: this function should be added later
        //std::shared_ptr<pdb::RamPointerBase> LazyAllocationHandler(void* pointer, size_t size);

    public:

        // raw pointer and the dimension for the vector
        std::vector<std::pair<page_id_t, std::size_t> > inputPages;

        std::vector<std::pair<page_id_t, std::size_t> > outputPages;

        std::vector<std::pair<float*, std::vector<size_t> >> inputArguments;

        std::pair<float*, std::vector<size_t> > outputArguments;

        float* copyBackArgument;

        cublasHandle_t cudaHandle;
        cudaStream_t cudaStream;

        PDBCUDAOpType op = PDBCUDAOpType::VectorAdd;

        CUDAStreamManager* stream_instance;
        CUDAStaticStorage* sstore_instance;
        CUDAMemMgr* memmgr_instance;
    };

}
#endif