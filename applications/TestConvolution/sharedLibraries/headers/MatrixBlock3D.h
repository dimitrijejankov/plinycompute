#pragma once

#include <Object.h>
#include <InterfaceFunctions.h>
#include "MatrixBlockMeta3D.h"
#include "MatrixBlockData3D.h"

namespace pdb::matrix_3d {

/**
 * This represents a cube in a large 3D distributed matrix.
 * For example if the large matrix has the size of 10000x10000x10000 and is split into 4 blocks of size 2500x2500x2500
 * Then we would have the following blocks in the system
 *
 * | metaData.x_id | metaData.y_id | metaData.z_id | data.x_size | data.y_size | data.z_size | data.block  |
 * |       0       |       1       |       0       |     25k     |     25k     |     25k     |  25k * 25k  |
 * |       1       |       1       |       0       |     25k     |     25k     |     25k     |  25k * 25k  |
 * |       0       |       0       |       0       |     25k     |     25k     |     25k     |  25k * 25k  |
 * |       1       |       0       |       0       |     25k     |     25k     |     25k     |  25k * 25k  |
 * |       0       |       1       |       1       |     25k     |     25k     |     25k     |  25k * 25k  |
 * |       1       |       1       |       1       |     25k     |     25k     |     25k     |  25k * 25k  |
 * |       0       |       0       |       1       |     25k     |     25k     |     25k     |  25k * 25k  |
 * |       1       |       0       |       1       |     25k     |     25k     |     25k     |  25k * 25k  |
 */

class MatrixBlock3D : public pdb::Object {
 public:

  /**
   * The default constructor
   */
  MatrixBlock3D() = default;

  /**
   * The constructor for a block size
   * @param x_id - the id of the block along the x axis
   * @param y_id - the id of the block along the y axis
   * @param z_id - the id of the block along the z axis
   * @param x_size - the size of the block along the x axis
   * @param y_size - the size of the block along the y axis
   * @param z_size - the size of the block along the z axis
   */
  MatrixBlock3D(uint32_t x_id, uint32_t y_id, uint32_t z_id, uint32_t x_size, uint32_t y_size, uint32_t z_size) {
    metaData = makeObject<MatrixBlockMeta3D>(x_id, y_id, z_id),
    data = makeObject<MatrixBlockData3D>(x_size, y_size, z_size);
  }

  ENABLE_DEEP_COPY

  /**
   * The metadata of the matrix
   */
   Handle<MatrixBlockMeta3D> metaData;

  /**
   * The data of the matrix
   */
  Handle<MatrixBlockData3D> data;

  /**
   *
   * @return
   */
  Handle<MatrixBlockMeta3D>& getKey() {
    return metaData;
  }

  /**
   *
   * @return
   */
  MatrixBlockMeta3D& getKeyRef(){
    return *metaData;
  }

  /**
   *
   * @return
   */
  Handle<MatrixBlockData3D>& getValue() {
    return data;
  }

  MatrixBlockData3D& getValueRef() {
    return *data;
  }

  uint32_t get_x_id() {
    return metaData->x_id;
  }

  uint32_t get_y_id() {
    return metaData->y_id;
  }

  uint32_t get_z_id() {
    return metaData->z_id;
  }

  uint32_t get_x_size() {
    return data->x_size;
  }

  uint32_t get_y_size() {
    return data->y_size;
  }

  uint32_t get_z_size() {
    return data->y_size;
  }
};

}
