/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/
#pragma once

#include "DoubleVector.h"
#include "Lambda.h"
#include "LambdaCreationFunctions.h"
#include "SelectionComp.h"
#include "FFMatrixBlock.h"
#include "FFSparseBlock.h"
#include <cstdlib>
#include <mkl_vsl.h>
#include <ctime>
#include <mkl.h>

namespace pdb {

// the sub namespace
namespace ff {

// FFSelectionSparsify will
class FFSelectionSparsify : public SelectionComp<FFSparseBlock, FFMatrixBlock> {

public:

  ENABLE_DEEP_COPY

  FFSelectionSparsify() = default;

  // srand has already been invoked in server
  Lambda<bool> getSelection(Handle<FFMatrixBlock> checkMe) override {
    return makeLambda(checkMe, [&](Handle<FFMatrixBlock> &checkMe) { return true; });
  }

  Lambda<Handle<FFSparseBlock>> getProjection(Handle<FFMatrixBlock> checkMe) override {
    return makeLambda(checkMe, [&](Handle<FFMatrixBlock> &checkMe) {

      auto numRows = checkMe->getNumRows();
      auto numCols = checkMe->getNumCols();
      auto data = checkMe->data->data->c_ptr();

      // allocate the room for the cols, rows and vals
      std::vector<MKL_INT> rows;
      rows.reserve(1000); // update this to make sense if necessary

      std::vector<MKL_INT> cols;
      rows.reserve(1000); // update this to make sense if necessary

      std::vector<float> vals;
      vals.reserve(1000); // update this to make sense if necessary

      // find all the non zero elements in the block
      MKL_INT nnz = 0;
      for(int32_t i = 0; i < numRows; ++i) {

        // generate the columns
        for(int32_t j = 0; j < numCols; ++j) {

          // do we have a non zero value
          if(data[i * numCols + j] != 0.0f) {

            // set the values
            rows.emplace_back(i);
            cols.emplace_back(j);
            vals.emplace_back(data[i * numCols + j]);

            nnz++;
          }
        }
      }

      // create the coo version of the matrix
      sparse_matrix_t cooMatrix;
      mkl_sparse_s_create_coo(&cooMatrix, SPARSE_INDEX_BASE_ZERO, numRows, numCols, nnz, rows.data(), cols.data(), vals.data());

      // convert the csr
      sparse_matrix_t csrMatrix;
      mkl_sparse_convert_csr(cooMatrix, SPARSE_OPERATION_NON_TRANSPOSE, &csrMatrix);

      // we store out values here
      sparse_index_base_t indexing;
      MKL_INT outRows;
      MKL_INT outCols;
      MKL_INT *rows_start;
      MKL_INT *rows_end;
      MKL_INT *col_indx;
      float *values;

      // get all the stuff out
      mkl_sparse_s_export_csr(csrMatrix, &indexing, &outRows, &outCols, &rows_start, &rows_end, &col_indx, &values);
      nnz = rows_end[outRows - 1] - rows_start[0];

      // make an output
      Handle<FFSparseBlock> out = makeObject<FFSparseBlock>(checkMe->getRowID(), checkMe->getColID(), outRows, outCols, nnz);

      // copy the matrix to the object
      memcpy(out->getValue()->rowIndices.c_ptr(), rows_start, sizeof(MKL_INT) * outRows);
      out->getValue()->rowIndices.c_ptr()[outRows] = rows_end[outRows - 1];
      memcpy(out->getValue()->values.c_ptr(), values, sizeof(float) * nnz);
      memcpy(out->getValue()->colIndices.c_ptr(), col_indx, sizeof(MKL_INT) * nnz);

      // destroy the matrix
      mkl_sparse_destroy(csrMatrix);
      mkl_sparse_destroy(cooMatrix);

      return out;
    });
  }
};
}
}