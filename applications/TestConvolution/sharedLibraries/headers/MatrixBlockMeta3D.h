#pragma once

#include <Object.h>
#include <Handle.h>

namespace pdb::matrix_3d {

class MatrixBlockMeta3D : public pdb::Object {
 public:

  /**
   * The default constructor
   */
  MatrixBlockMeta3D() = default;

  MatrixBlockMeta3D(uint32_t x_id, uint32_t y_id, uint32_t z_id) : x_id(x_id), y_id(y_id), z_id(z_id) {}

  ENABLE_DEEP_COPY

  // the id of the block along the x axis
  uint32_t x_id;

  // the id of the block along the y axis
  uint32_t y_id;

  // the id of the block along the z axis
  uint32_t z_id;

  // get the y_id above this block
  pdb::Handle<MatrixBlockMeta3D> above() { return pdb::makeObject<MatrixBlockMeta3D>(x_id, y_id + 1, z_id); }

  // get the y_id below this block
  pdb::Handle<MatrixBlockMeta3D>  below()  { return pdb::makeObject<MatrixBlockMeta3D>(x_id, y_id - 1, z_id); }

  // get the x_id of the block left to this one
  pdb::Handle<MatrixBlockMeta3D>  left()  { return pdb::makeObject<MatrixBlockMeta3D>(x_id - 1, y_id, z_id); }

  // get the x_id of the block right of this one
  pdb::Handle<MatrixBlockMeta3D>  right()  { return pdb::makeObject<MatrixBlockMeta3D>(x_id + 1, y_id, z_id);}

  // get the z_id of the block to the front of this one
  pdb::Handle<MatrixBlockMeta3D>  front()  { return pdb::makeObject<MatrixBlockMeta3D>(x_id, y_id, z_id + 1); }

  // get the z_id of the block to the back of this one
  pdb::Handle<MatrixBlockMeta3D>  back()  { return pdb::makeObject<MatrixBlockMeta3D>(x_id, y_id, z_id - 1); }

  bool operator==(const MatrixBlockMeta3D &other) const {
    return x_id == other.x_id && y_id == other.y_id && z_id == other.z_id;
  }

  size_t hash() const {
    return 1024 * 1024 * z_id + 1024 * y_id + x_id;
  }
};

}