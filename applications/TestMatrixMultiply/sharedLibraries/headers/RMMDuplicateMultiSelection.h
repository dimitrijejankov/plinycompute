#pragma once
#include "Lambda.h"
#include "LambdaCreationFunctions.h"
#include "TRABlock.h"
#include "MultiSelectionComp.h"
#include <DeepCopy.h>

namespace pdb {
namespace matrix {
class RMMDuplicateMultiSelection : public MultiSelectionComp<TRABlock, TRABlock> {
public:
    ENABLE_DEEP_COPY

    RMMDuplicateMultiSelection() {}

    RMMDuplicateMultiSelection(uint32_t newKeyDim, uint32_t duplicateCount):
    newKeyDim_(newKeyDim), duplicateCount_(duplicateCount) {}

    Lambda<bool> getSelection(Handle<TRABlock> checkMe) override {
        return makeLambda(checkMe, [](Handle<TRABlock>& checkMe) { return true; });
    }

    Lambda<Vector<Handle<TRABlock>>> getProjection(Handle<TRABlock> checkMe) override {
        return makeLambda(checkMe, [&](Handle<TRABlock>& checkMe) { return this->insertDim(checkMe); });
    }

private:
    uint32_t newKeyDim_;
    uint32_t duplicateCount_;

    //Todo Binhang: the construction function see the comment there.
    Vector<Handle<TRABlock>> insertDim(Handle<TRABlock> checkme){
        Vector<Handle<TRABlock>> result;
        for(uint32_t i =0; i< duplicateCount_; i++){
            if (newKeyDim_ == 0){
                Handle<TRABlock> current_block = makeObject<TRABlock>(i, checkme->getkey0(), checkme->getkey1(),
                                                                      checkme->getValue());
                result.push_back(current_block);
            }
            else if (newKeyDim_ == 2){
                Handle<TRABlock> current_block = makeObject<TRABlock>(checkme->getkey0(), checkme->getkey0(), i,
                                                                      checkme->getValue());
                result.push_back(current_block);
            }
        }
        return result;
    }
};
}
}