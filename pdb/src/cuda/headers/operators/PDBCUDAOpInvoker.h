#ifndef PDB_CUDA_INVOKER
#define PDB_CUDA_INVOKER

#include <iostream>
#include <Handle.h>
#include <vector>
#include "PDBVector.h"
#include "PDBCUDAOpType.h"
#include "PDBBufferManagerImpl.h"
#include "utility/PDBCUDAUtility.h"
#include "storage/PDBCUDAMemoryManager.h"

// simply support two kind of operations
namespace pdb{

class PDBCUDAInvoker{
public:
    PDBCUDAInvoker() = default;

    void setInput(float* input, std::vector<size_t>& inputDim);

    void setOutput(float* output, std::vector<size_t>& outputDim);

    bool invoke();
};
}
#endif