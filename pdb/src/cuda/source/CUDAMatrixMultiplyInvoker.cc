#include "CUDAMatrixMultiplyInvoker.h"
#include "CUDAStreamManager.h"

extern void* gpuMemoryManager;
extern void* gpuStreamManager;
extern void* gpuStaticStorage;
extern void* gpuDynamicStorage;

namespace pdb {

    PDBCUDAMatrixMultipleInvoker::PDBCUDAMatrixMultipleInvoker() {
        sstore_instance = static_cast<CUDAStaticStorage*>(gpuStaticStorage);
        memmgr_instance = static_cast<CUDAMemMgr*>(gpuMemoryManager);
        stream_instance = static_cast<CUDAStreamManager*>(gpuStreamManager);
        PDBCUDAStreamUtils util = stream_instance->bindCPUThreadToStream();
        cudaStream = util.first;
        cudaHandle = util.second;
    }

    PDBCUDAMatrixMultipleInvoker::~PDBCUDAMatrixMultipleInvoker(){
        for (auto pageID : inputPages){
            memmgr_instance->UnpinPageImpl(pageID.first,true);
        }
        for (auto pageID : outputPages){
            memmgr_instance->UnpinPageImpl(pageID.first, true);
        }
    }

    void PDBCUDAMatrixMultipleInvoker::setInput(float* input, const std::vector<size_t> &inputDim) {
        int isDevice = isDevicePointer((void *) input);
        if (isDevice) {
            inputArguments.push_back(std::make_pair(input, inputDim));
            return;
        }
        // get CPU page for this object
        auto cpuPageInfo = sstore_instance->getCPUPageFromObjectAddress((void*)input);

        page_id_t cudaPageID;
        // get GPU page based on CPU page information
        CUDAPage* cudaPage = sstore_instance->getGPUPageFromCPUPage(cpuPageInfo, &cudaPageID);

        // if page is never written, move the content from CPU page to GPU page.
        // Notice, here, the size of GPU page may be larger than CPU page. Some smart way for De-fragmentation is needed.
        if (!cudaPage->isMoved()){
            checkCudaErrors(cudaMemcpyAsync(cudaPage->getBytes(), cpuPageInfo.first, cpuPageInfo.second, cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStream));
            cudaPage->setIsMoved(true);
        }

        // get the object address on GPU page
        void* cudaObjectPointer = static_cast<char*>(cudaPage->getBytes()) + sstore_instance->getObjectOffsetWithCPUPage(cpuPageInfo.first, input);
        inputArguments.push_back(std::make_pair(static_cast<float*> (cudaObjectPointer), inputDim));

        // book keep the page id and the real number of bytes used
        inputPages.push_back(std::make_pair(cudaPageID, cpuPageInfo.second));
    }

    void PDBCUDAMatrixMultipleInvoker::setOutput(float* output, const std::vector<size_t> &outputDim) {

        int isDevice = isDevicePointer((void *) output);
        if (isDevice) {
            outputArguments.first = output;
            outputArguments.second = outputDim;
            return;
        }
        auto cpuPageInfo = sstore_instance->getCPUPageFromObjectAddress(static_cast<void*>(output));

        page_id_t cudaPageID;

        CUDAPage* cudaPage = sstore_instance->getGPUPageFromCPUPage(cpuPageInfo, &cudaPageID);

        void* cudaObjectPointer = static_cast<char*>(cudaPage->getBytes()) + sstore_instance->getObjectOffsetWithCPUPage(cpuPageInfo.first, output);

        outputArguments = std::make_pair(static_cast<float *>(cudaObjectPointer), outputDim);

        outputPages.push_back(std::make_pair(cudaPageID, cpuPageInfo.second));

        // TODO: this should be removed in the future.
        copyBackArgument = output;
    }

    bool PDBCUDAMatrixMultipleInvoker::invoke() {

        //std::cout << (long) pthread_self() << " :PDBCUDAMatrixMultipleInvoker invoke() \n";
        kernel(inputArguments[0].first, inputArguments[1].first, outputArguments.first, inputArguments[0].second[0],
                      inputArguments[0].second[1], inputArguments[1].second[0]);
        return true;
    }

    void
    PDBCUDAMatrixMultipleInvoker::kernel(float *in1data, float *in2data, float *outdata, size_t in1NumRow, size_t in1NumCol,
                                                size_t in2NumCol) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasErrCheck( cublasSgemm(cudaHandle, CUBLAS_OP_N, CUBLAS_OP_N, in1NumRow, in2NumCol, in1NumCol, &alpha, in1data, in1NumRow,
                                    in2data, in1NumCol, &beta, outdata, in1NumRow));

        // TODO: this should be removed in the future.
        checkCudaErrors(cudaMemcpyAsync(static_cast<void*>(copyBackArgument), static_cast<void*>(outputArguments.first), outputArguments.second[0] * outputArguments.second[1] * sizeof(float), cudaMemcpyDeviceToHost, cudaStream));
    }
}


