#pragma once

#include <LambdaCreationFunctions.h>
#include <mkl_cblas.h>
#include "JoinComp.h"
#include "TensorBlock.h"

namespace pdb::matrix {

    //Todo:: Change this to local join
    class RMMJoin : public JoinComp <RMMJoin, TensorBlock, TensorBlock, TensorBlock> {
    public:

        ENABLE_DEEP_COPY

        RMMJoin() = default;

        static Lambda <bool> getKeySelection (Handle <TensorBlockMeta> in1, Handle <TensorBlockMeta> in2) {
            return (makeLambdaFromMethod (in1, getIdx0) == makeLambdaFromMethod (in2, getIdx0))
            and (makeLambdaFromMethod (in1, getIdx1) == makeLambdaFromMethod (in2, getIdx1))
            and (makeLambdaFromMethod (in1, getIdx2) == makeLambdaFromMethod (in2, getIdx2));
        }

        static Lambda <Handle<TensorBlockMeta>> getKeyProjection(Handle <TensorBlockMeta> in1, Handle <TensorBlockMeta> in2) {
            return makeLambda (in1, in2, [] (Handle <TensorBlockMeta> &in1, Handle <TensorBlockMeta> &in2) {
                Handle<TensorBlockMeta> out = makeObject<TensorBlockMeta>(in1->getIdx0(), in1->getIdx1(), in1->getIdx2());
                return out;
            });
        }

        static Lambda <Handle<TensorBlockData>> getValueProjection(Handle <TensorBlockData> in1, Handle <TensorBlockData> in2) {
            return makeLambda (in1, in2, [] (Handle <TensorBlockData> &in1, Handle <TensorBlockData> &in2) {

                // get the sizes
                uint32_t I = in1->dim0;
                uint32_t J = in2->dim1;
                uint32_t K = in1->dim1;

                // make an output
                Handle<TensorBlockData> out = makeObject<TensorBlockData>(I, J, 1);

                // get the ptrs
                float *outData = out->data->c_ptr();
                float *in1Data = in1->data->c_ptr();
                float *in2Data = in2->data->c_ptr();

                // do the multiply
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, J, K, 1.0f, in1Data, K, in2Data, J, 0.0f, outData, J);

                // return the output
                return out;
            });
        }
    };
}