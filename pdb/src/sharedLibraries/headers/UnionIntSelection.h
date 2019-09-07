#pragma once

#include <Employee.h>
#include <Supervisor.h>
#include <SelectionComp.h>
#include "LambdaCreationFunctions.h"

namespace pdb {

class UnionIntSelection : public SelectionComp<int, int> {
private:

  // the reminder of division by 3 we want to keep
  int reminder = 0;

public:

  UnionIntSelection() = default;

  explicit UnionIntSelection(int reminder) : reminder(reminder) {}

  ENABLE_DEEP_COPY

  Lambda<bool> getSelection(Handle<int> checkMe) override {
    return makeLambda(checkMe, [&](Handle<int>& checkMe) {
      return *checkMe % 3 == reminder;
    });
  }

  Lambda<Handle<int>> getProjection(Handle<int> checkMe) override{
    return makeLambda(checkMe, [&](Handle<int>& checkMe) {
      Handle<int> result = makeObject<int>();
      *result = *checkMe;
      return result;
    });
  }

};

}