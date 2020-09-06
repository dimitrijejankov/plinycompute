#pragma once
#include "Lambda.h"
#include "LambdaCreationFunctions.h"
#include "TensorBlock.h"
#include "MultiSelectionComp.h"
#include <DeepCopy.h>

namespace pdb {
namespace matrix {
class RMMDuplicateMultiSelection : public MultiSelectionComp<TensorBlock, TensorBlock> {
public:
    ENABLE_DEEP_COPY

    RMMDuplicateMultiSelection() {}

    RMMDuplicateMultiSelection(uint32_t newKeyDim, uint32_t duplicateCount):
    newKeyDim_(newKeyDim), duplicateCount_(duplicateCount) {}

    Lambda<bool> getSelection(Handle<TensorBlock> checkMe) override {
        return makeLambda(checkMe, [](Handle<TensorBlock>& checkMe) { return true; });
    }

    Lambda<Vector<Handle<TensorBlock>>> getProjection(Handle<TensorBlock> checkMe) override {
        return makeLambda(checkMe, [&](Handle<TensorBlock>& checkMe) { return this->insertDim(checkMe); });
    }

private:
    uint32_t newKeyDim_;
    uint32_t duplicateCount_;

    //Todo Binhang: the construction function see the comment there.
    Vector<Handle<TensorBlock>> insertDim(Handle<TensorBlock> checkme){
        Vector<Handle<TensorBlock>> result;
        for(uint32_t i =0; i<duplicateCount_; i++){
            if (newKeyDim_ == 0){
                Handle<TensorBlock> current_block = makeObject<TensorBlock>(i, checkme->getkey0(),checkme->getDim1(), checkme->getValue());
                result.push_back(current_block);
            }
            else if (newKeyDim_ == 2){
                Handle<TensorBlock> current_block = makeObject<TensorBlock>(checkme->getkey0(),checkme->getDim1(), i, checkme->getValue());
                result.push_back(current_block);
            }
        }
        return result;
    }
};
}
}