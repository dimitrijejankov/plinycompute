#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <mkl_types.h>

namespace pdb {

// the sub namespace
namespace ff {

class FFSparseBlockData : public pdb::Object {
public:

  /**
   * The default constructor
   */
  FFSparseBlockData() = default;

  FFSparseBlockData(MKL_INT rowID, MKL_INT colID, MKL_INT numRows, MKL_INT numCols, size_t nnz) : rowID(rowID),
                                                                                                  colID(colID),
                                                                                                  numRows(numRows),
                                                                                                  numCols(numCols),
                                                                                                  rowIndices(numRows + 1, numRows + 1),
                                                                                                  colIndices(nnz, nnz),
                                                                                                  values(nnz, nnz) {}

  ENABLE_DEEP_COPY

  // the number of rows the strip has
  MKL_INT rowID = 0;

  // the number of columns the strip has
  MKL_INT colID = 0;

  // the number of rows the strip has
  MKL_INT numRows = 0;

  // the number of columns the strip has
  MKL_INT numCols = 0;

  // sparse matrix stuff
  pdb::Vector<MKL_INT> rowIndices;
  pdb::Vector<MKL_INT> colIndices;
  pdb::Vector<float> values;
};

}

}