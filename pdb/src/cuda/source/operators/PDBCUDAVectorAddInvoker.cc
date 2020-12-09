#include <assert.h>
#include "operators/PDBCUDAVectorAddInvoker.h"
#include "stream/PDBCUDAStreamManager.h"

extern void* gpuMemoryManager;
extern void* gpuStreamManager;
extern void* gpuStaticStorage;
extern void* gpuDynamicStorage;
extern std::atomic<int> debugger;

namespace pdb {
    PDBCUDAVectorAddInvoker::PDBCUDAVectorAddInvoker() {

        sstore_instance = static_cast<PDBCUDAStaticStorage*>(gpuStaticStorage);
        memmgr_instance = static_cast<PDBCUDAMemoryManager*>(gpuMemoryManager);
        stream_instance = static_cast<PDBCUDAStreamManager*>(gpuStreamManager);

        PDBCUDAStreamUtils util = stream_instance->bindCPUThreadToStream();

        cudaStream = util.first;
        cudaHandle = util.second;
    }

    PDBCUDAVectorAddInvoker::~PDBCUDAVectorAddInvoker() {

        for (auto pageID : inputPages){
            memmgr_instance->UnpinPageImpl(pageID.first,true);
        }
        for (auto pageID: outputPages){
            memmgr_instance->UnpinPageImpl(pageID.first, true);
        }
    }

    bool PDBCUDAVectorAddInvoker::invoke() {
        kernel(outputArguments.first, inputArguments[0].first, inputArguments[0].second[0]);
        return true;
    }

    /**
     * Perform SAXPY on vector elements: outdata[] = outdata[] + in1data[];
     * @param in2data
     * @param in2data
     * @param in1data
     * @param N
     */
    void PDBCUDAVectorAddInvoker::kernel(float *in1data, float *in2data, size_t N) {
        const float alpha = 1.0;
        cublasErrCheck(cublasSaxpy(cudaHandle, N, &alpha, in2data, 1, in1data, 1));
        //TODO:
        cudaMemcpyAsync(static_cast<void*>(copyBackArgument), static_cast<void*>(outputArguments.first), outputArguments.second[0] * sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
        cudaError_t err = cudaGetLastError();
        if (err!=cudaSuccess){
            throw std::runtime_error("cuda Error!\n");
        }
    }

    void PDBCUDAVectorAddInvoker::setInput(float *input, const std::vector<size_t> &inputDim) {

        int isDevice = isDevicePointer((void *) input);
        if (isDevice) {
            inputArguments.push_back(std::make_pair(input, inputDim));
            return;
        }

        // get CPU page for this object
        auto cpuPageInfo = sstore_instance->getCPUPageFromObjectAddress(static_cast<void*>(input));

        page_id_t  cudaPageID;
        // get GPU page based on CPU page information
        PDBCUDAPage* cudaPage = sstore_instance->getGPUPageFromCPUPage(cpuPageInfo, &cudaPageID);

        // fetch GPU page
        // if page is never written, move the content from CPU page to GPU page.
        // Notice, here, the size of GPU page may be larger than CPU page. Some smart way for De-fragmentation is needed.
        if (!cudaPage->isMoved()){
            checkCudaErrors(cudaMemcpyAsync(cudaPage->getBytes(), cpuPageInfo.first, cpuPageInfo.second, cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStream));
            cudaPage->setIsMoved(true);
        }

        void* cudaObjectPointer = static_cast<char*>(cudaPage->getBytes()) + sstore_instance->getObjectOffsetWithCPUPage(cpuPageInfo.first, input);
        std::cout << "setInput Argument: " << cudaObjectPointer << std::endl;
        inputArguments.push_back(std::make_pair(static_cast<float*> (cudaObjectPointer), inputDim));

        // book keep the page id and the real number of bytes used
        inputPages.push_back(std::make_pair(cudaPageID, cpuPageInfo.second));
    }

    // std::shared_ptr<pdb::RamPointerBase> PDBCUDAVectorAddInvoker::LazyAllocationHandler(void* pointer, size_t size){
    //    pair<void *, size_t> PageInfo = (static_cast<PDBCUDAMemoryManager *>(gpuMemoryManager))->getObjectCPUPage(
    //            (void *) pointer);
    //    return (static_cast<PDBCUDAMemoryManager *>(gpuMemoryManager))->handleInputObjectWithRamPointer(PageInfo, (void*)pointer, size, cudaStream);
    // }

    void PDBCUDAVectorAddInvoker::setOutput(float *output, const std::vector<size_t> & outputDim) {

        debugger = debugger+1;

        int isDevice = isDevicePointer((void *) output);
        if (isDevice) {
            outputArguments.first = output;
            outputArguments.second = outputDim;
            return;
        }

        auto cpuPageInfo = sstore_instance->getCPUPageFromObjectAddress(static_cast<void*>(output));

        page_id_t cudaPageID;

        PDBCUDAPage* cudaPage = sstore_instance->getGPUPageFromCPUPage(cpuPageInfo, &cudaPageID);

        void* cudaObjectPointer = static_cast<char*>(cudaPage->getBytes()) + sstore_instance->getObjectOffsetWithCPUPage(cpuPageInfo.first, output);

        std::cout << "setOutput Argument: " << cudaObjectPointer << std::endl;

        if (cudaObjectPointer == reinterpret_cast<void*>(0x17c)){
            exit(0);
        }

        outputArguments = std::make_pair(static_cast<float *>(cudaObjectPointer), outputDim);

        outputPages.push_back(std::make_pair(cudaPageID, cpuPageInfo.second));

        // TODO: this should be removed in the future.
        copyBackArgument = output;
    }
};