#include "PDBCUDAGPUInvoke.h"
#include "storage/PDBRamPointer.h"

/** SimpleTypeGPUInvoke deals with all the primitive types and invoke the gpu kernel for the input/output
 * `Out` vector should be reserved before passing as parameter
 * @tparam InvokerType - operator type (should be a derived type from PDBCUDAOp)
 * @tparam InputType - trivial copyable type for input params
 * @tparam OutputType - trivial copyable type for output params
 * @param f - operator
 * @param Out - output param
 * @param OutDim - output param dimension
 * @param In1 - input param 1
 * @param In1Dim - input param 1 dimension
 * @param In2 - input param 2
 * @param In2Dim - input param 2 dimension
 * @return bool - successful or not
 */
template<typename InvokerType, typename InputType, typename OutputType>
typename std::enable_if_t<
        is_base_of<pdb::PDBCUDAInvoker, InvokerType>::value && std::is_trivially_copyable<OutputType>::value &&
        std::is_trivially_copyable<InputType>::value, bool>
SimpleTypeGPUInvoke(InvokerType &f, OutputType *Out, std::vector<size_t> &OutDim, InputType *In1,
                    std::vector<size_t> &In1Dim, InputType *In2, std::vector<size_t> &In2Dim) {
    f.setInput(In1, In1Dim);
    f.setInput(In2, In2Dim);
    f.setOutput(Out, OutDim);
    bool res = f.invoke();
    return res;
}

/** SimpleTypeGPUInvoke deals with all the primitive types and invoke the gpu kernel for the input/output
 * `Out` vector should be reserved before passing as parameter
 * @tparam InvokerType - operator type (should be a derived type from PDBCUDAOp)
 * @tparam InputType - trivial copyable type for input params
 * @tparam OutputType - trivial copyable type for output params
 * @param f - operator
 * @param Out - output param
 * @param OutDim - output param dimension
 * @param In1 - input param 1
 * @param In1Dim - input param 1 dimension
 * @return bool - successful or not
 */
template<typename InvokerType, typename InputType, typename OutputType>
typename std::enable_if_t<
        is_base_of<pdb::PDBCUDAInvoker, InvokerType>::value && std::is_trivially_copyable<OutputType>::value &&
        std::is_trivially_copyable<InputType>::value, bool>
SimpleTypeGPUInvoke(InvokerType &f, OutputType *Out, std::vector<size_t> &OutDim, InputType *In1,
                    std::vector<size_t> &In1Dim) {
    f.setInput(In1, In1Dim);
    f.setOutput(Out, OutDim);
    bool res = f.invoke();
    return res;
}

/**
 * GPUInvoke for handling the case that both input/output param is Handle<SimpleType>
 * @tparam InvokerType - operator type (should be a derived type from PDBCUDAOp)
 * @tparam InputType - trivial copyable type for input params
 * @tparam OutputType - trivial copyable type for output params
 * @param f - operator
 * @param Out - output param
 * @param In1 - input param 1
 * @param In2 - input param 2
 * @return bool - successful or not
 */
template<typename InvokerType, typename InputType, typename OutputType>
typename std::enable_if_t<is_base_of<pdb::PDBCUDAInvoker, InvokerType>::value, bool>
GPUInvoke(InvokerType &f, pdb::Handle<OutputType> Output, pdb::Handle<InputType> In1, pdb::Handle<InputType> In2) {
    auto In1Object = (In1.getTarget())->getObject();
    std::vector<size_t> In1Dim{1};
    auto In2Object = (In2.getTarget())->getObject();
    std::vector<size_t> In2Dim{1};
    auto OutputObject = (Output.getTarget())->getObject();
    std::vector<size_t> OutDim{1};
    return SimpleTypeGPUInvoke(f, OutputObject, OutDim, In1Object, In1Dim, In2Object, In2Dim);
}


/**
 * GPUInvoke for handling the case that dimensions for all the input/output params are 1 dimensional array
 * @tparam InvokerType - operator type (should be a derived type from PDBCUDAOp)
 * @tparam InputType - trivial copyable type for input params
 * @tparam OutputType - trivial copyable type for output params
 * @param f - operator
 * @param Out - output param
 * @param In1 - input param 1
 * @param In2 - input param 2
 * @return bool - successful or not
 */
template<typename InvokerType, typename InputType, typename OutputType>
typename std::enable_if_t<is_base_of<pdb::PDBCUDAInvoker, InvokerType>::value, bool>
GPUInvoke(InvokerType &f, pdb::Handle<pdb::Vector<OutputType>> Out, pdb::Handle<pdb::Vector<InputType>> In1,
          pdb::Handle<pdb::Vector<InputType>> In2) {
    auto In1Object = In1->c_ptr();
    std::vector<size_t> In1Dim{In1->size()};
    auto In2Object = In2->c_ptr();
    std::vector<size_t> In2Dim{In2->size()};
    auto OutObject = Out->c_ptr();
    std::vector<size_t> OutDim{Out->size()};
    return SimpleTypeGPUInvoke(f, OutObject, OutDim, In1Object, In1Dim, In2Object, In2Dim);
}

/**
 * GPUInvoke just allow trivial copyable types. Handle 1 input param case.
 * @tparam InvokerType - operator type (should be a derived type from PDBCUDAOp)
 * @tparam InputType - trivial copyable type for input params
 * @tparam OutputType - trivial copyable type for output params
 * @param f - operator
 * @param Out - output param
 * @param OutDim - output param dimension
 * @param In1 - input param 1
 * @param In1Dim - input param 1 dimension
 * @return bool - successful or not
 *
*/
template<typename InvokerType, typename InputType, typename OutputType>
typename std::enable_if_t<is_base_of<pdb::PDBCUDAInvoker, InvokerType>::value, bool>
GPUInvoke(InvokerType &f, pdb::Handle<pdb::Vector<OutputType>> Out, std::vector<size_t> &OutDim,
          pdb::Handle<pdb::Vector<InputType>> In1, std::vector<size_t> &In1Dim) {
    auto In1Object = In1->c_ptr();
    auto OutObject = Out->c_ptr();
    return SimpleTypeGPUInvoke(f, OutObject, OutDim, In1Object, In1Dim);
}


/**
 * GPUInvoke just allow trivial copyable types. Handle 2 input param case.
 * @tparam InvokerType - operator type (should be a derived type from PDBCUDAOp)
 * @tparam InputType - trivial copyable type for input params
 * @tparam OutputType - trivial copyable type for output params
 * @param f - operator
 * @param Out - output param
 * @param OutDim - output param dimension
 * @param In1 - input param 1
 * @param In1Dim - input param 1 dimension
 * @param In2 - input param 2
 * @param In2Dim - input param 2 dimension
 * @return bool - successful or not
 */
template<typename InvokerType, typename InputType, typename OutputType>
typename std::enable_if_t<is_base_of<pdb::PDBCUDAInvoker, InvokerType>::value, bool>
GPUInvoke(InvokerType &f, pdb::Handle<pdb::Vector<OutputType>> Out, std::vector<size_t> &OutDim,
          pdb::Handle<pdb::Vector<InputType>> In1, std::vector<size_t> &In1Dim,
          pdb::Handle<pdb::Vector<InputType> > In2, std::vector<size_t> &In2Dim) {
    auto In1Object = In1->c_ptr();
    auto In2Object = In2->c_ptr();
    auto OutObject = Out->c_ptr();
    return SimpleTypeGPUInvoke(f, OutObject, OutDim, In1Object, In1Dim, In2Object, In2Dim);
}

bool GPUInvoke(pdb::PDBCUDAMatrixMultipleInvoker &f, pdb::Handle<pdb::Vector<float>> Out, std::vector<size_t> &OutDim,
               pdb::Handle<pdb::Vector<float>> In1, std::vector<size_t> &In1Dim, pdb::Handle<pdb::Vector<float> > In2,
               std::vector<size_t> &In2Dim) {
    auto In1Object = In1->c_ptr();
    auto In2Object = In2->c_ptr();
    auto OutObject = Out->c_ptr();
    return SimpleTypeGPUInvoke(f, OutObject, OutDim, In1Object, In1Dim, In2Object, In2Dim);
}

bool GPUInvoke(pdb::PDBCUDAVectorAddInvoker &f, pdb::Handle<pdb::Vector<float>> Out, std::vector<size_t> &OutDim,
               pdb::Handle<pdb::Vector<float>> In1, std::vector<size_t> &In1Dim) {
    auto In1Object = In1->c_ptr();
    auto OutObject = Out->c_ptr();
    return SimpleTypeGPUInvoke(f, OutObject, OutDim, In1Object, In1Dim);
}

//TODO: this should be added later
/*
std::shared_ptr<pdb::RamPointerBase>
GPULazyAllocationHandler(pdb::PDBCUDAVectorAddInvoker &f, void* pointer, size_t size) {
    return f.LazyAllocationHandler(pointer, size);
}
 */


/** By default, this GPUInvoke will handle the matrix multiple case for join.
 * @param op
 * @param Out
 * @param OutDim
 * @param In1
 * @param In1Dim
 * @return
 */

//TODO: this should be added back later
/*
bool GPUInvoke(pdb::PDBCUDAOpType &op, pdb::Handle<pdb::Vector<float>> Out, std::vector<size_t> &OutDim,
               pdb::Handle<pdb::Vector<float>> In1, std::vector<size_t> &In1Dim) {
    if (op != pdb::PDBCUDAOpType::VectorAdd) {
        exit(-1);
    }
    pdb::PDBCUDAVectorAddInvoker vectorAddInvoker;

    auto OutPtr = Out->c_ptr();
    auto OutCPUPtr = Out->cpu_ptr();
    bool onGPU = Out->onGPU();

    auto In1Ptr = In1->c_ptr();

    // For handling the case of lazy allocation

    // In1Ptr == In1CPUPtr.
    // means the situation that pointer address in cpu ram is equal to pointer address in gpu/cpu ram.
    // This situation has two cases: 1. data should not on GPU. 2. data should on GPU but is lazy allocated.
    // We check the onGPU flag to see which case.
    if (OutCPUPtr == OutPtr && onGPU){
        std::shared_ptr<pdb::RamPointerBase> NewRamPointer = GPULazyAllocationHandler(vectorAddInvoker, static_cast<void*>(OutCPUPtr), OutDim[0]);
        Out->setRamPointerReference(NewRamPointer);
        OutPtr = Out->c_ptr();
    }
    assert(In1Ptr != nullptr);
    assert(OutPtr != nullptr);

    return SimpleTypeGPUInvoke(vectorAddInvoker, OutPtr, OutDim, In1Ptr, In1Dim);
}
 */

/** By default, this GPUInvoke will handle the vector add case for aggregation.
 * @param op
 * @param Out
 * @param OutDim
 * @param In1
 * @param In1Dim
 * @param In2
 * @param In2Dim
 * @return
 */

bool GPUInvoke(pdb::PDBCUDAOpType &op, pdb::Handle<pdb::Vector<float>> Out, std::vector<size_t> &OutDim,
               pdb::Handle<pdb::Vector<float>> In1, std::vector<size_t> &In1Dim, pdb::Handle<pdb::Vector<float> > In2,
               std::vector<size_t> &In2Dim) {
    if (op != pdb::PDBCUDAOpType::MatrixMultiple) {
        exit(-1);
    }
    pdb::PDBCUDAMatrixMultipleInvoker matrixMultipleInvoker;
    auto In1Object = In1->c_ptr();
    auto In2Object = In2->c_ptr();
    auto OutObject = Out->c_ptr();
    return SimpleTypeGPUInvoke(matrixMultipleInvoker, OutObject, OutDim, In1Object, In1Dim, In2Object, In2Dim);
}

bool GPUInvoke(pdb::PDBCUDAOpType& op, pdb::Handle<pdb::Vector<float>> Out, std::vector<size_t>& OutDim, pdb::Handle<pdb::Vector<float>> In1, std::vector<size_t>& In1Dim){
    if (op!=pdb::PDBCUDAOpType::VectorAdd){
        exit(-1);
    }
    pdb::PDBCUDAVectorAddInvoker vectorAddInvoker;
    auto In1Object = In1->c_ptr();
    auto OutObject = Out->c_ptr();
    return SimpleTypeGPUInvoke(vectorAddInvoker, OutObject, OutDim, In1Object, In1Dim);
}
