#pragma once

#include "LambdaCreationFunctions.h"
#include "SelectionComp.h"
#include "MatrixBlock.h"

namespace pdb {

class MatrixIdentitySelection : public SelectionComp<MatrixIdentitySelection, pdb::matrix::MatrixBlock, pdb::matrix::MatrixBlock> {

public:

  ENABLE_DEEP_COPY

  MatrixIdentitySelection() = default;

  Lambda<bool> getSelection(Handle<pdb::matrix::MatrixBlock> checkMe) {
    return makeLambda(checkMe, [&](Handle<pdb::matrix::MatrixBlock> &checkMe) {
      return true;
    });
  }

  Lambda<Handle<pdb::matrix::MatrixBlock>>
  static getProjection(Handle<pdb::matrix::MatrixBlock> checkMe) {
    return makeLambda(checkMe, [](Handle<pdb::matrix::MatrixBlock> &checkMe) {
      return checkMe;
    });
  }
};

}
