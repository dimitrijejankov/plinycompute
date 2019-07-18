#pragma once

#include <Object.h>
#include <PDBVector.h>

namespace pdb {

// the sub namespace
namespace matrix {

class MatrixBlockData : public pdb::Object {
public:

  /**
   * The default constructor
   */
  MatrixBlockData() = default;

  MatrixBlockData(uint32_t numRows, uint32_t numCols) : numRows(numRows), numCols(numCols) {

    // allocate the data
    data = makeObject<Vector<float>>(numRows * numCols, numRows * numCols);
  }

  ENABLE_DEEP_COPY

  /**
   * The number of rows in the block
   */
  uint32_t numRows = 0;

  /**
   * The number of columns in the block
   */
  uint32_t numCols = 0;

  /**
   * The values of the block
   */
  Handle<Vector<float>> data;

  /**
   * Does the summation of the data
   * @param other - the other
   * @return
   */
  MatrixBlockData& operator+(MatrixBlockData& other) {

    // get the data
    float *myData = data->c_ptr();
    float *otherData = other.data->c_ptr();

    // sum up the data
    for (int i = 0; i < numRows * numCols; i++) {
      (myData)[i] += (otherData)[i];
    }

    // return me
    return *this;
  }
};

}

}
