#ifndef PDB_CUDA_MM_INVOKER
#define PDB_CUDA_MM_INVOKER

#include <iostream>
#include <Handle.h>
#include <functional>
#include <numeric>
#include <utility>
#include "PDBVector.h"
#include "CUDAUtility.h"
#include "CUDAOpType.h"
#include "CUDAOpInvoker.h"
#include "CUDAStaticStorage.h"
#include "CUDAStreamManager.h"

// simply support two kind of operations
namespace pdb {

    /**
    * PDBCUDAMatrixMultipleInvoker - A wrapper for cublas sgemm
    */
    class PDBCUDAMatrixMultipleInvoker : public PDBCUDAInvoker {
    public:

        PDBCUDAMatrixMultipleInvoker();

        ~PDBCUDAMatrixMultipleInvoker();

        bool invoke();

        void kernel(float* in1data, float* in2data, float *outdata, size_t in1NumRow, size_t in1NumCol, size_t in2NumCol);

        void setInput(float *input, const std::vector<size_t> &inputDim);

        void setOutput(float *output, const std::vector<size_t> &outputDim);

    public:

        std::vector<std::pair<float*, std::vector<size_t> >> inputArguments;

        // Book-keep all the input pages and the actually bytes used.
        std::vector<std::pair<page_id_t, size_t> > inputPages;

        std::vector<std::pair<page_id_t, size_t> > outputPages;

        //TODO: should make it into a vector
        std::pair<float*, std::vector<size_t> > outputArguments;

        float* copyBackArgument;

        cudaStream_t cudaStream;
        cublasHandle_t cudaHandle;

        CUDAStreamManager* stream_instance;
        CUDAStaticStorage* sstore_instance;
        CUDAMemoryManager* memmgr_instance;
    };

}
#endif