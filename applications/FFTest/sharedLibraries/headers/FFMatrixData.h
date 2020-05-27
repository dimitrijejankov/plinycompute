#pragma once

#include <Object.h>
#include <PDBVector.h>

namespace pdb {

// the sub namespace
namespace ff {

class FFMatrixData : public pdb::Object {
 public:

  /**
   * The default constructor
   */
  FFMatrixData() = default;

  FFMatrixData(uint32_t numRows, uint32_t numCols, uint32_t rowID, uint32_t colID) : numRows(numRows), numCols(numCols) {

    // allocate the data
    data = makeObject<Vector<float>>(numRows * numCols, numRows * numCols);
  }

  ENABLE_DEEP_COPY

  /**
   * The row id of the matrix
   */
  uint32_t rowID;

  /**
   * the column id of the matrix
   */
  uint32_t colID;

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
   * The values of the bias
   */
  Handle<Vector<float>> bias;

  /**
   * Does the summation of the data
   * @param other - the other
   * @return
   */
  FFMatrixData& operator+(FFMatrixData& other) {

    // get the data
    float *myData = data->c_ptr();
    float *otherData = other.data->c_ptr();

    // sum up the data
    for (int i = 0; i < numRows * numCols; i++) {
      (myData)[i] += (otherData)[i];
    }

    // sup up the bios if we need to
    if(bias != nullptr && other.bias != nullptr) {
      myData = bias->c_ptr();
      otherData = other.bias->c_ptr();
      for (int i = 0; i < other.bias->size(); i++) {
        (myData)[i] += (otherData)[i];
      }
    }

    // return me
    return *this;
  }
};

}

}