#pragma once

#include <LambdaCreationFunctions.h>
#include <operators/PDBCUDAOpType.h>
#include <PDBCUDAGPUInvoke.h>
#include "JoinComp.h"
#include "MatrixBlock.h"

namespace pdb::matrix {

class MatrixMultiplyJoin : public JoinComp <MatrixMultiplyJoin, MatrixBlock, MatrixBlock, MatrixBlock> {
 public:

  ENABLE_DEEP_COPY

  MatrixMultiplyJoin() = default;

  static Lambda <bool> getKeySelection (Handle <MatrixBlockMeta> in1, Handle <MatrixBlockMeta> in2) {
    return (makeLambdaFromMember (in1, colID) == makeLambdaFromMember (in2, rowID));
  }

  static Lambda <Handle<MatrixBlockMeta>> getKeyProjection(Handle <MatrixBlockMeta> in1, Handle <MatrixBlockMeta> in2) {
    return makeLambda (in1, in2, [] (Handle <MatrixBlockMeta> &in1, Handle <MatrixBlockMeta> &in2) {
      Handle<MatrixBlockMeta> out = makeObject<MatrixBlockMeta>(in1->rowID, in2->colID);
      return out;
    });
  }

  static Lambda <Handle<MatrixBlockData>> getValueProjection(Handle <MatrixBlockData> in1, Handle <MatrixBlockData> in2) {
    return makeLambda (in1, in2, [] (Handle <MatrixBlockData> &in1, Handle <MatrixBlockData> &in2) {

      // get the sizes
      uint32_t I = in1->numRows;
      uint32_t J = in2->numCols;
      uint32_t K = in1->numCols;

      // make an output
      Handle<MatrixBlockData> out = makeObject<MatrixBlockData>(I, J);

      vector<size_t> outdim = {I,J};
      vector<size_t> in1dim = {I,K};
      vector<size_t> in2dim = {K,J};
      pdb::PDBCUDAOpType op = pdb::PDBCUDAOpType::MatrixMultiple;
      GPUInvoke(op, out->data, outdim, in1->data, in1dim, in2->data, in2dim);
      // return the output
      return out;
    });
  }
};
}