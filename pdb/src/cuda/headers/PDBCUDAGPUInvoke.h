#ifndef PDB_CUDA_GPU_INVOKE
#define PDB_CUDA_GPU_INVOKE

#include <iostream>
#include "operators/PDBCUDAMatrixMultiplyInvoker.h"
#include "operators/PDBCUDAVectorAddInvoker.h"
#include <vector>

std::shared_ptr<pdb::RamPointerBase>
GPULazyAllocationHandler(pdb::PDBCUDAVectorAddInvoker &f, void* pointer, size_t size);

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
                    std::vector<size_t> &In1Dim, InputType *In2, std::vector<size_t> &In2Dim);

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
                    std::vector<size_t> &In1Dim);

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
GPUInvoke(InvokerType &f, pdb::Handle<OutputType> Output, pdb::Handle<InputType> In1, pdb::Handle<InputType> In2);

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
          pdb::Handle<pdb::Vector<InputType>> In2);

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
          pdb::Handle<pdb::Vector<InputType>> In1, std::vector<size_t> &In1Dim);


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
          pdb::Handle<pdb::Vector<InputType> > In2, std::vector<size_t> &In2Dim);

bool GPUInvoke(pdb::PDBCUDAMatrixMultipleInvoker &f, pdb::Handle<pdb::Vector<float>> Out, std::vector<size_t> &OutDim,
               pdb::Handle<pdb::Vector<float>> In1, std::vector<size_t> &In1Dim, pdb::Handle<pdb::Vector<float> > In2,
               std::vector<size_t> &In2Dim);

bool GPUInvoke(pdb::PDBCUDAVectorAddInvoker &f, pdb::Handle<pdb::Vector<float>> Out, std::vector<size_t> &OutDim,
               pdb::Handle<pdb::Vector<float>> In1, std::vector<size_t> &In1Dim);

/** By default, this GPUInvoke will handle the matrix multiple case for join.
 * @param op
 * @param Out
 * @param OutDim
 * @param In1
 * @param In1Dim
 * @return
 */

bool GPUInvoke(pdb::PDBCUDAOpType &op, pdb::Handle<pdb::Vector<float>> Out, std::vector<size_t> &OutDim,
               pdb::Handle<pdb::Vector<float>> In1, std::vector<size_t> &In1Dim);

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
               std::vector<size_t> &In2Dim);


#endif