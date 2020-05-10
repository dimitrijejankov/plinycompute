#pragma once

#include <LambdaCreationFunctions.h>
#include "Process.h"
#include "JoinComp.h"
#include "MatrixBlock3D.h"
#include "MatrixConvResult.h"

namespace pdb::matrix_3d {

#define get_value(A, X, Y, Z) (A[(X) + block_size * ((Y) + block_size * (Z))])
#define get_out(A, X, Y, Z) (A[(X) + x_out_size * ((Y) + y_out_size * (Z))])
#define get_conv_value(A, X, Y, Z) (A[(X) + 3 * ((Y) + 3 * (Z))])

class MatrixConv3DJoin : public JoinComp<MatrixConv3DJoin, MatrixConvResult, /** out **/
                                         MatrixBlock3D, /** in1 **/
                                         MatrixBlock3D, /** in2 **/
                                         MatrixBlock3D, /** in3 **/
                                         MatrixBlock3D, /** in4 **/
                                         MatrixBlock3D, /** in5 **/
                                         MatrixBlock3D, /** in6 **/
                                         MatrixBlock3D, /** in7 **/
                                         MatrixBlock3D> /** in8 **/ {
 public:

  ENABLE_DEEP_COPY

  MatrixConv3DJoin(uint32_t block_size_x, uint32_t block_size_y, uint32_t block_size_z, uint32_t num_channels) : block_size_x(block_size_x),
                                                                                                                 block_size_y(block_size_y),
                                                                                                                 block_size_z(block_size_z),
                                                                                                                 num_in_channels(num_channels) {}

  MatrixConv3DJoin() = default;

  // we match the top left front record to the rest
  static Lambda<bool> getKeySelection(Handle<MatrixBlockMeta3D> top_left_front, //
                                      Handle<MatrixBlockMeta3D> top_right_front, //
                                      Handle<MatrixBlockMeta3D> bottom_left_front, //
                                      Handle<MatrixBlockMeta3D> bottom_right_front, //
                                      Handle<MatrixBlockMeta3D> top_left_back, // in0
                                      Handle<MatrixBlockMeta3D> top_right_back, //
                                      Handle<MatrixBlockMeta3D> bottom_left_back, //
                                      Handle<MatrixBlockMeta3D> bottom_right_back) { //  in1

    return (makeLambdaFromMethod (top_left_front, right) == makeLambdaFromSelf(top_right_front)) &&
           (makeLambdaFromMethod (top_left_front, below) == makeLambdaFromSelf(bottom_left_front)) &&
           (makeLambdaFromMethod (top_left_front, right) == makeLambdaFromMethod (bottom_right_front, above)) &&
           (makeLambdaFromMethod (top_left_front, back) == makeLambdaFromSelf(top_left_back)) &&
           (makeLambdaFromMethod (top_left_back, right) == makeLambdaFromSelf(top_right_back)) &&
           (makeLambdaFromMethod (top_left_back, below) == makeLambdaFromSelf(bottom_left_back)) &&
           (makeLambdaFromMethod (top_left_back, right) == makeLambdaFromMethod (bottom_right_back, above));
  }

  Lambda<Handle<MatrixBlockMeta3D>> getKeyProjection(Handle<MatrixBlockMeta3D> top_left_front,
                                                     Handle<MatrixBlockMeta3D> top_right_front,
                                                     Handle<MatrixBlockMeta3D> bottom_left_front,
                                                     Handle<MatrixBlockMeta3D> bottom_right_front,
                                                     Handle<MatrixBlockMeta3D> top_left_back,
                                                     Handle<MatrixBlockMeta3D> top_right_back,
                                                     Handle<MatrixBlockMeta3D> bottom_left_back,
                                                     Handle<MatrixBlockMeta3D> bottom_right_back) {
    return makeLambda(top_left_front,
                      top_right_front,
                      bottom_left_front,
                      bottom_right_front,
                      top_left_back,
                      top_right_back,
                      bottom_left_back,
                      bottom_right_back, [](Handle<MatrixBlockMeta3D> &top_left_front,
                                            Handle<MatrixBlockMeta3D> &top_right_front,
                                            Handle<MatrixBlockMeta3D> &bottom_left_front,
                                            Handle<MatrixBlockMeta3D> &bottom_right_front,
                                            Handle<MatrixBlockMeta3D> &top_left_back,
                                            Handle<MatrixBlockMeta3D> &top_right_back,
                                            Handle<MatrixBlockMeta3D> &bottom_left_back,
                                            Handle<MatrixBlockMeta3D> &bottom_right_back) {
          Handle<MatrixBlockMeta3D> out =
              makeObject<MatrixBlockMeta3D>(top_left_front->x_id, top_left_front->y_id, top_left_front->z_id);
          return out;
        });
  }

  Lambda<Handle<Vector<MatrixBlockData3D>>> getValueProjection(Handle<MatrixBlockData3D> top_left_front,
                                                               Handle<MatrixBlockData3D> top_right_front,
                                                               Handle<MatrixBlockData3D> bottom_left_front,
                                                               Handle<MatrixBlockData3D> bottom_right_front,
                                                               Handle<MatrixBlockData3D> top_left_back,
                                                               Handle<MatrixBlockData3D> top_right_back,
                                                               Handle<MatrixBlockData3D> bottom_left_back,
                                                               Handle<MatrixBlockData3D> bottom_right_back) {
    return makeLambda(top_left_front,
                      top_right_front,
                      bottom_left_front,
                      bottom_right_front,
                      top_left_back,
                      top_right_back,
                      bottom_left_back,
                      bottom_right_back, [&](Handle<MatrixBlockData3D> &top_left_front,
                                                     Handle<MatrixBlockData3D> &top_right_front,
                                                     Handle<MatrixBlockData3D> &bottom_left_front,
                                                     Handle<MatrixBlockData3D> &bottom_right_front,
                                                     Handle<MatrixBlockData3D> &top_left_back,
                                                     Handle<MatrixBlockData3D> &top_right_back,
                                                     Handle<MatrixBlockData3D> &bottom_left_back,
                                                     Handle<MatrixBlockData3D> &bottom_right_back) {

          int my_x_offset = block_size_x / 2;
          int my_y_offset = block_size_y / 2;
          int my_z_offset = block_size_z / 2;

          int my_x_boundary = block_size_x / 2;
          int my_y_boundary = block_size_y / 2;
          int my_z_boundary = block_size_z / 2;

          auto x_in_size = block_size_x;
          auto y_in_size = block_size_y;
          auto z_in_size = block_size_z;

          // if the left ones are on the left boundary
          if (top_left_front->isLeftBorder) {
            x_in_size += block_size_x / 2;
            my_x_offset -= block_size_x / 2;
          }

          // if the right ones are the the right boundary
          if (top_right_front->isRightBorder) {
            x_in_size += block_size_x / 2;
            my_x_boundary -= block_size_x / 2;
          }

          // if the top ones are on the top boundary
          if (top_left_front->isTopBorder) {
            y_in_size += block_size_y / 2;
            my_y_offset -= block_size_y / 2;
          }

          // if the bottom ones are on the bottom boundary
          if (bottom_left_front->isBottomBorder) {
            y_in_size += block_size_y / 2;
            my_y_boundary -= block_size_y / 2;
          }

          // figure out the boundary for the z axis
          if (bottom_left_front->isFrontBorder) {
            z_in_size += block_size_z / 2;
            my_z_offset -= block_size_z / 2;
          }

          // figure out the boundary for the z axis
          if (bottom_left_back->isBackBorder) {
            z_in_size += block_size_z / 2;
            my_z_boundary -= block_size_z / 2;
          }

          // make an output
          Handle<Vector<MatrixBlockData3D>> out = makeObject<Vector<MatrixBlockData3D>>(1, 1);

          // init the filters
          (*out)[0] = MatrixBlockData3D(x_in_size - 3 + 1, y_in_size - 3 + 1, z_in_size - 3 + 1, num_out_channels);

          process(top_left_front->data->c_ptr(),
                  top_right_front->data->c_ptr(),
                  bottom_left_front->data->c_ptr(),
                  bottom_right_front->data->c_ptr(),
                  top_left_back->data->c_ptr(),
                  top_right_back->data->c_ptr(),
                  bottom_left_back->data->c_ptr(),
                  bottom_right_back->data->c_ptr(),
                  (*out)[0].data->c_ptr(),
                  block_size_x,
                  block_size_y,
                  block_size_z,
                  x_in_size,
                  y_in_size,
                  z_in_size,
                  my_x_offset,
                  my_y_offset,
                  my_z_offset,
                  my_x_boundary,
                  my_y_boundary,
                  my_z_boundary,
                  3,
                  3,
                  3,
                  num_in_channels,
                  num_out_channels);

          // return the output
          return out;
        });
  }

  uint32_t block_size_x;
  uint32_t block_size_y;
  uint32_t block_size_z;
  uint32_t num_out_channels = 5;
  uint32_t num_in_channels = 3;
};

}

#undef get_value
#undef get_out
#undef get_conv_value