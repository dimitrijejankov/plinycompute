#pragma once

#include <Object.h>
#include <InterfaceFunctions.h>
#include "MatrixBlockMeta3D.h"
#include "MatrixBlockData3D.h"

namespace pdb::matrix_3d {


class MatrixConvResult : public pdb::Object {
 public:

  /**
   * The default constructor
   */
  MatrixConvResult() = default;

  MatrixConvResult(int32_t x_id, int32_t y_id, int32_t z_id, int32_t x_size, int32_t y_size, int32_t z_size, uint32_t numChannels) {
    metaData = makeObject<MatrixBlockMeta3D>(x_id, y_id, z_id);
    channels = makeObject<Vector<MatrixBlockData3D>>(numChannels, numChannels);
    for(int i = 0; i < numChannels; ++i) {
      (*channels)[i] = MatrixBlockData3D(x_size, y_size, z_size);
    }
  }

  ENABLE_DEEP_COPY

  /**
   * The metadata of the matrix
   */
  Handle<MatrixBlockMeta3D> metaData;

  /**
   * there are multiple channels
   */
  Handle<Vector<MatrixBlockData3D>> channels;

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
  Handle<Vector<MatrixBlockData3D>>& getValue() {
    return channels;
  }

  Vector<MatrixBlockData3D>& getValueRef() {
    return *channels;
  }

};

}