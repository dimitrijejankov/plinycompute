#pragma once

#include <Object.h>
#include <Handle.h>
#include <PDBVector.h>
#include <mkl_types.h>
#include "FFMatrixMeta.h"
#include "FFSparseBlockData.h"

namespace pdb {

// the sub namespace
namespace ff {

class FFSparseBlock : public pdb::Object {
public:

  /**
   * The default constructor
   */
  FFSparseBlock() = default;

  /**
   * The constructor for a strip
   * @param rowID
   * @param colID
   * @param numRows - the number of rows the block has
   * @param numCols - the number of columns the block has
   * @param nnr - the size of the rowIndex vector
   * @param nnz - the number of the non zero values
   */
  FFSparseBlock(uint32_t rowID, uint64_t colID, MKL_INT numRows, MKL_INT numCols, size_t nnz) {
    metaData = makeObject<FFMatrixMeta>(rowID, colID),
    data = makeObject<FFSparseBlockData>(rowID, colID, numRows, numCols, nnz);
  }

  ENABLE_DEEP_COPY

  /**
   * The metadata of the matrix
   */
  Handle<FFMatrixMeta> metaData;

  /**
   * The data of the matrix
   */
  Handle<FFSparseBlockData> data;
  /**
   *
   * @return
   */
  Handle<FFMatrixMeta>& getKey() {
    return metaData;
  }

  /**
   *
   * @return
   */
  FFMatrixMeta& getKeyRef(){
    return *metaData;
  }

  /**
   *
   * @return
   */
  Handle<FFSparseBlockData>& getValue() {
    return data;
  }

  FFSparseBlockData& getValueRef() {
    return *data;
  }

  uint32_t getRowID() {
    return metaData->rowID;
  }

  uint32_t getColID() {
    return metaData->colID;
  }

  uint32_t getNumRows() {
    return data->numRows;
  }

  uint32_t getNumCols() {
    return data->numCols;
  }
};

}

}