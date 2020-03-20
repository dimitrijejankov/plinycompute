#pragma once

#include <LambdaCreationFunctions.h>
#include <mkl_cblas.h>
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

  MatrixConv3DJoin(uint32_t x_left_boundary,
                   uint32_t x_right_boundary,
                   uint32_t y_top_boundary,
                   uint32_t y_bottom_boundary,
                   uint32_t z_front_boundary,
                   uint32_t z_back_boundary,
                   uint32_t block_size) : x_left_boundary(x_left_boundary),
                                          x_right_boundary(x_right_boundary),
                                          y_top_boundary(y_top_boundary),
                                          y_bottom_boundary(y_bottom_boundary),
                                          z_front_boundary(z_front_boundary),
                                          z_back_boundary(z_back_boundary),
                                          block_size(block_size) {}

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

  inline void apply(const float *a, float *out, const float *filter) {

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          *out += get_value(a, i, j, k) * get_conv_value(filter, i, j, k);
        }
      }
    }
  }

  void convolve(const float *a,
                int my_x_offset, int my_y_offset, int my_z_offset,
                float *out, float *filter,
                int x_boundary, int y_boundary, int z_boundary,
                int x_out_size, int y_out_size, int z_out_size) {

    for (int i = my_x_offset; i < x_boundary - 2; ++i) {
      for (int j = my_y_offset; j < y_boundary - 2; ++j) {
        for (int k = my_z_offset; k < z_boundary - 2; ++k) {
          apply(&get_value(a, i, j, k),
                &get_out(out, i - my_x_offset, j - my_y_offset, k - my_z_offset),
                filter);
        }
      }
    }
  }

  void doFilter(float *top_left_front,
                float *top_right_front,
                float *bottom_left_front,
                float *bottom_right_front,
                float *top_left_back,
                float *top_right_back,
                float *bottom_left_back,
                float *bottom_right_back,
                float *out,
                float *filter,
                int32_t x_out_size,
                int32_t y_out_size,
                int32_t z_out_size,
                int32_t my_x_offset,
                int32_t my_y_offset,
                int32_t my_z_offset,
                int32_t my_x_boundary,
                int32_t my_y_boundary,
                int32_t my_z_boundary) {

    // -------------------------------------------------TOP-------------------------------------------------------------

    // perform convolution on top_left_front [ok]
    convolve(top_left_front,
             my_x_offset, my_y_offset, my_z_offset,
             &get_out(out, 0, 0, 0),
             filter,
             block_size, block_size, block_size,
             x_out_size, y_out_size, z_out_size);

    // perform convolution on top_right_front [ok]
    convolve(top_right_front,
             0, my_y_offset, my_z_offset,
             &get_out(out, block_size - my_x_offset, 0, 0),
             filter, my_x_boundary, block_size, block_size,
             x_out_size, y_out_size, z_out_size);

    // perform convolution on the top left back [ok]
    convolve(top_left_back,
             my_x_offset, my_y_offset, 0,
             &get_out(out, 0, 0, block_size - my_z_offset),
             filter, block_size, block_size, my_z_boundary,
             x_out_size, y_out_size, z_out_size);

    // perform convolution on the top right back
    convolve(top_right_back,
             0, my_y_offset, 0,
             &get_out(out, block_size - my_x_offset, 0, block_size - my_z_offset),
             filter, my_x_boundary, block_size, my_z_boundary,
             x_out_size, y_out_size, z_out_size);

    // ------------------------------------------------BOTTOM-----------------------------------------------------------

    // perform convolution on bottom_left_front [ok]
    convolve(bottom_left_front,
             my_x_offset, 0, my_z_offset,
             &get_out(out, 0, block_size - my_y_offset, 0),
             filter, block_size, my_y_boundary, block_size,
             x_out_size, y_out_size, z_out_size);

    // perform convolution on bottom_right_front [ok]
    convolve(bottom_right_front, 0, 0, my_z_offset,
             &get_out(out, block_size - my_x_offset, block_size - my_y_offset, 0),
             filter, my_x_boundary, my_y_boundary, block_size,
             x_out_size, y_out_size, z_out_size);

    // perform convolution on bottom top left back [ok]
    convolve(bottom_left_back,
             my_x_offset, 0, 0,
             &get_out(out, 0, block_size - my_y_offset, block_size - my_z_offset),
             filter, block_size, my_y_boundary, my_z_boundary,
             x_out_size, y_out_size, z_out_size);

    // perform convolution on bottom top right back
    convolve(bottom_right_back,
             0, 0, 0,
             &get_out(out, block_size - my_x_offset, block_size - my_y_offset, block_size - my_z_offset),
             filter, my_x_boundary, my_y_boundary, my_z_boundary,
             x_out_size, y_out_size, z_out_size);

    // -----------------------------------------------BETWEEN-----------------------------------------------------------
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

          int my_x_offset = block_size / 2;
          int my_y_offset = block_size / 2;
          int my_z_offset = block_size / 2;

          int my_x_boundary = block_size / 2;
          int my_y_boundary = block_size / 2;
          int my_z_boundary = block_size / 2;

          auto x_out_size = block_size - my_x_offset + my_x_boundary - 2;
          auto y_out_size = block_size - my_y_offset + my_y_boundary - 2;
          auto z_out_size = block_size - my_z_offset + my_z_boundary - 2;

          // if the left ones are on the left boundary
          if (top_left_front->isLeftBorder) {
            x_out_size += block_size / 2;
            my_x_offset -= block_size / 2;
          }

          // if the right ones are the the right boundary
          if (top_right_front->isRightBorder) {
            x_out_size += block_size / 2;
            my_x_boundary += block_size / 2;
          }

          // if the top ones are on the top boundary
          if (top_left_front->isTopBorder) {
            y_out_size += block_size / 2;
            my_y_offset -= block_size / 2;
          }

          // if the bottom ones are on the bottom boundary
          if (bottom_left_front->isBottomBorder) {
            y_out_size += block_size / 2;
            my_y_boundary += block_size / 2;
          }

          // figure out the boundary for the z axis
          if (bottom_left_front->isFrontBorder) {
            z_out_size += block_size / 2;
            my_z_offset -= block_size / 2;
          }

          // figure out the boundary for the z axis
          if (bottom_left_front->isBackBorder) {
            y_out_size += block_size / 2;
            my_y_boundary += block_size / 2;
          }

          // make an output
          Handle<Vector<MatrixBlockData3D>> out = makeObject<Vector<MatrixBlockData3D>>(3, 3);

          // init the filters
          (*out)[0] = MatrixBlockData3D(x_out_size, y_out_size, z_out_size);
          (*out)[1] = MatrixBlockData3D(x_out_size, y_out_size, z_out_size);
          (*out)[2] = MatrixBlockData3D(x_out_size, y_out_size, z_out_size);

          /// 1. Do the first filter

          doFilter(top_left_front->data->c_ptr(),
                   top_right_front->data->c_ptr(),
                   bottom_left_front->data->c_ptr(),
                   bottom_right_front->data->c_ptr(),
                   top_left_back->data->c_ptr(),
                   top_right_back->data->c_ptr(),
                   bottom_left_back->data->c_ptr(),
                   bottom_right_back->data->c_ptr(),
                   (*out)[0].data->c_ptr(),
                   sobel_x,
                   x_out_size,
                   y_out_size,
                   z_out_size,
                   my_x_offset,
                   my_y_offset,
                   my_z_offset,
                   my_x_boundary,
                   my_y_boundary,
                   my_z_boundary);

          /// 2. Do the second filter

          doFilter(top_left_front->data->c_ptr(),
                   top_right_front->data->c_ptr(),
                   bottom_left_front->data->c_ptr(),
                   bottom_right_front->data->c_ptr(),
                   top_left_back->data->c_ptr(),
                   top_right_back->data->c_ptr(),
                   bottom_left_back->data->c_ptr(),
                   bottom_right_back->data->c_ptr(),
                   (*out)[1].data->c_ptr(),
                   sobel_y,
                   x_out_size,
                   y_out_size,
                   z_out_size,
                   my_x_offset,
                   my_y_offset,
                   my_z_offset,
                   my_x_boundary,
                   my_y_boundary,
                   my_z_boundary);

          /// 2. Do the third filter

          doFilter(top_left_front->data->c_ptr(),
                   top_right_front->data->c_ptr(),
                   bottom_left_front->data->c_ptr(),
                   bottom_right_front->data->c_ptr(),
                   top_left_back->data->c_ptr(),
                   top_right_back->data->c_ptr(),
                   bottom_left_back->data->c_ptr(),
                   bottom_right_back->data->c_ptr(),
                   (*out)[2].data->c_ptr(),
                   sobel_z,
                   x_out_size,
                   y_out_size,
                   z_out_size,
                   my_x_offset,
                   my_y_offset,
                   my_z_offset,
                   my_x_boundary,
                   my_y_boundary,
                   my_z_boundary);

          // return the output
          return out;
        });
  }

  // y is changed every three, z every 9
  float sobel_x[27] = {1.0f, 0.0f, 1.0f,
                       3.0f, 0.0f, 3.0f,
                       1.0f, 0.0f, 1.0f,

                       3.0f, 0.0f, 3.0f,
                       6.0f, 0.0f, 6.0f,
                       3.0f, 0.0f, 3.0f,

                       1.0f, 0.0f, 1.0f,
                       3.0f, 0.0f, 3.0f,
                       1.0f, 0.0f, 1.0f};

  float sobel_y[27] = {1.0f, 3.0f, 1.0f,
                       0.0f, 0.0f, 0.0f,
                       1.0f, 3.0f, 1.0f,

                       3.0f, 6.0f, 3.0f,
                       0.0f, 0.0f, 0.0f,
                       3.0f, 6.0f, 3.0f,

                       1.0f, 3.0f, 1.0f,
                       0.0f, 0.0f, 0.0f,
                       1.0f, 3.0f, 1.0f};

  float sobel_z[27] = {1.0f, 3.0f, 1.0f,
                       3.0f, 6.0f, 3.0f,
                       1.0f, 3.0f, 1.0f,

                       0.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 0.0f,

                       1.0f, 3.0f, 1.0f,
                       3.0f, 6.0f, 3.0f,
                       1.0f, 3.0f, 1.0f};

  uint32_t x_left_boundary;
  uint32_t x_right_boundary;
  uint32_t y_top_boundary;
  uint32_t y_bottom_boundary;
  uint32_t z_front_boundary;
  uint32_t z_back_boundary;
  uint32_t block_size;
};

}

#undef get_value
#undef get_out
#undef get_conv_value