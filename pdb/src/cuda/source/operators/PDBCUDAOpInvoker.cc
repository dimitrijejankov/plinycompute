#include <iostream>
#include "operators/PDBCUDAOpInvoker.h"

namespace pdb {
    void PDBCUDAInvoker::setInput(float *input, std::vector<size_t> &inputDim) {
        std::cerr << "There is no setInput() method for base class\n";
        exit(-1);
    }
    void PDBCUDAInvoker::setOutput(float *output, std::vector<size_t> &outputDim) {
        std::cerr << "There is no setOutput() method for base class\n";
        exit(-1);
    }
    bool PDBCUDAInvoker::invoke() {
        std::cerr << "There is no invoke() method for base class\n";
        exit(-1);
    }
}