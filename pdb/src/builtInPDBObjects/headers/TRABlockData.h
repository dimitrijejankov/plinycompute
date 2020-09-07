#pragma once

//  PRELOAD %TRABlockData%

#include <Object.h>
#include <PDBVector.h>

namespace pdb {

class TRABlockData : public pdb::Object {
 public:

  /**
   * The default constructor
   */
  TRABlockData() = default;

  TRABlockData(uint32_t dim0, uint32_t dim1, uint32_t dim2) : dim0(dim0), dim1(dim1), dim2(dim2) {

    // allocate the data
    data = makeObject<Vector<float>>(dim0 * dim1 * dim2, dim0 * dim1 * dim2);
  }

  ENABLE_DEEP_COPY

  // The number of rows in the block if interpreted as matrix
  uint32_t dim0 = 1;

  // The number of columns in the block if interpreted as matrix
  uint32_t dim1 = 1;

  uint32_t dim2 = 1;

  /**
   * The values of the block
   */
  Handle<Vector<float>> data;

  /**
   * Does the summation of the data
   * @param other - the other
   * @return
   */
  TRABlockData& operator+(TRABlockData& other) {

    // get the data
    float *myData = data->c_ptr();
    float *otherData = other.data->c_ptr();

    // sum up the data
    for (int i = 0; i < dim0 * dim1; i++) {
      (myData)[i] += (otherData)[i];
    }

    // return me
    return *this;
  }
};

}
