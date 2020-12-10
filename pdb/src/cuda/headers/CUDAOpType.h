#ifndef PDB_CUDA_OP_TYPE
#define PDB_CUDA_OP_TYPE

#include <iostream>
#include <Handle.h>
#include "PDBVector.h"
#include "CUDAUtility.h"
namespace pdb {
    enum PDBCUDAOpType {
        SimpleAdd,
        SimpleMultiple,
        MatrixMultiple, //0
        VectorAdd, //1
    };
}
#endif