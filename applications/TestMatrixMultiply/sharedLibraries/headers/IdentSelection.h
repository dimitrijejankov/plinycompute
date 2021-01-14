#pragma once

#include "TRABlock.h"
#include <SelectionComp.h>
#include "LambdaCreationFunctions.h"

namespace pdb {

class IdentSelection : public SelectionComp<TRABlock, TRABlock> {
 private:

  // the reminder of division by 3 we want to keep
  int reminder = 0;

 public:

  IdentSelection() = default;

  explicit IdentSelection(int reminder) : reminder(reminder) {}

  ENABLE_DEEP_COPY

  Lambda<bool> getSelection(Handle<TRABlock> checkMe) override {
    return makeLambda(checkMe, [&](Handle<TRABlock>& checkMe) {
      return true;
    });
  }

  Lambda<Handle<TRABlock>> getProjection(Handle<TRABlock> checkMe) override{
    return makeLambda(checkMe, [&](Handle<TRABlock>& checkMe) {
      Handle<TRABlock> result = makeObject<TRABlock>();
      *result = *checkMe;
      return result;
    });
  }

};

}