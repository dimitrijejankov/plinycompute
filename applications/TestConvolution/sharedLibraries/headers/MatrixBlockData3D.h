#pragma once

#include <Object.h>
#include <PDBVector.h>

namespace pdb::matrix_3d {

class MatrixBlockData3D : public pdb::Object {
 public:

  /**
   * The default constructor
   */
  MatrixBlockData3D() = default;

  MatrixBlockData3D(uint32_t x_size, uint32_t y_size, uint32_t z_size, uint32_t channels) : x_size(x_size), y_size(y_size), z_size(z_size) {

    // allocate the data
    data = makeObject<Vector<float>>(x_size * y_size * z_size * channels, x_size * y_size * z_size * channels);
  }

  MatrixBlockData3D(uint32_t x_size, uint32_t y_size, uint32_t z_size) : x_size(x_size), y_size(y_size), z_size(z_size) {

    // allocate the data
    data = makeObject<Vector<float>>(x_size * y_size * z_size, x_size * y_size * z_size);
  }

  ENABLE_DEEP_COPY

  // the size of the block along the x axis
  uint32_t x_size;

  // the size of the bloc along the y axis
  uint32_t y_size;

  // the size of the block along the z axis
  uint32_t z_size;

  // the values of the block
  Handle<Vector<float>> data;

  bool isLeftBorder{false};
  bool isRightBorder{false};
  bool isTopBorder{false};
  bool isBottomBorder{false};
  bool isFrontBorder{false};
  bool isBackBorder{false};

  /**
   * Does the summation of the data
   * @param other - the other
   * @return
   */
  MatrixBlockData3D& operator+(MatrixBlockData3D& other) {

    // get the data
    float *myData = data->c_ptr();
    float *otherData = other.data->c_ptr();

    // sum up the data
    for (int i = 0; i < x_size * y_size * z_size; i++) {
      (myData)[i] += (otherData)[i];
    }

    // return me
    return *this;
  }
};

}