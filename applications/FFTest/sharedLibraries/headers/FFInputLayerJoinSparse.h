#pragma once

#include <Object.h>
#include <Handle.h>
#include <LambdaCreationFunctions.h>
#include <JoinComp.h>
#include <mkl.h>
#include "FFMatrixMeta.h"
#include "FFSparseBlock.h"
#include "FFMatrixData.h"
#include "FFMatrixBlock.h"

namespace pdb {

// the sub namespace
namespace ff {

class FFInputLayerJoinSparse : public JoinComp <FFInputLayerJoinSparse, FFMatrixBlock, FFSparseBlock, FFMatrixBlock> {
 public:

  ENABLE_DEEP_COPY

  FFInputLayerJoinSparse() = default;

  static Lambda <bool> getKeySelection (Handle <FFMatrixMeta> in1, Handle <FFMatrixMeta> in2) {
    return (makeLambdaFromMember (in1, colID) == makeLambdaFromMember (in2, rowID));
  }

  static Lambda <Handle<FFMatrixMeta>> getKeyProjection(Handle <FFMatrixMeta> in1, Handle <FFMatrixMeta> in2) {
    return makeLambda (in1, in2, [] (Handle <FFMatrixMeta> &in1, Handle <FFMatrixMeta> &in2) {
      Handle<FFMatrixMeta> out = makeObject<FFMatrixMeta>(in1->rowID, in2->colID);
      return out;
    });
  }

  static Lambda <Handle<FFMatrixData>> getValueProjection(Handle <FFSparseBlockData> in1, Handle <FFMatrixData> in2) {
    return makeLambda (in1, in2, [] (Handle <FFSparseBlockData> &in1, Handle <FFMatrixData> &in2) {

      // get the sizes
      uint32_t I = in1->numRows;
      uint32_t J = in2->numCols;
      uint32_t K = in1->numCols;

      // make an output
      Handle<FFMatrixData> out = makeObject<FFMatrixData>(I, J, in1->rowID, in2->colID);

      // need to convert this to sc
      sparse_matrix_t csrA = nullptr;

      // create the csr matrix
      mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, in1->numRows, in1->numCols,
                              in1->rowIndices.c_ptr(), in1->rowIndices.c_ptr() + 1,
                              in1->colIndices.c_ptr(), in1->values.c_ptr());

      // this descriptor is kinda necessary
      struct matrix_descr descrA;
      descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

      // do the multiply
      auto outData = out->data->c_ptr();
      mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, // A is not transposed
                      1.0f, // A is not scaled
                      csrA, // give the A matrix
                      descrA, // this is a general matrix
                      SPARSE_LAYOUT_ROW_MAJOR, // even though this says sparse it relates to the dense matrix...
                      in2->data->c_ptr(), // this is the data of the dense matrix in2
                      J, // number of columns in the output matrix
                      J, // since in2 is row major dimension in2.numCols is the leading dimension
                      0.0f,  // the output matrix is not added to the result
                      outData, // the output matrix
                      J); // has the same number of columns as in2

      // process the bias if necessary
      if(in2->bias != nullptr) {
        auto bias = in2->bias->c_ptr();
        for(uint32_t r = 0; r < I; r++) {
          for(uint32_t c = 0; c < J; c++) {

            // add the bias
            outData[r * J + c] += bias[c];
          }
        }
      }

      // return the output
      return out;
    });
  }
};

}

}