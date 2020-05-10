#include "memory.h"
#include "../../sharedLibraries/headers/Process.h"
#include <ATen/ATen.h>
#include <ATen/Functions.h>


#define get_value(A, X, Y, Z) (A[(X) + x_size * ((Y) + y_size * (Z))])
#define get_out(A, X, Y, Z) (A[(X) + x_out_stride * ((Y) + y_out_stride * (Z))])


namespace pdb {

void write_to_tensor(float *out, int x_out_stride, int y_out_stride, int z_out_stride,
                     float *in, int x_pos, int y_pos, int z_pos, int x_size, int y_size, int z_size, int input_channels) {

  // move into position
  out = &get_out(out, x_pos, y_pos, z_pos);

  // go through all channels
  for(int c = 0; c < input_channels; ++c){

    // copy the tensor channel here
    for(int z = 0; z < z_size; ++z) {
      for(int y = 0; y < y_size; ++y) {
        for(int x = 0; x < x_size; ++x) {
          get_out(out, x, y, z) = get_value(in, x, y, z);
        }
      }
    }

    // move to next channel
    out = out + (x_out_stride * y_out_stride * z_out_stride);
  }
}

void _process(float *top_left_front,
              float *top_right_front,
              float *bottom_left_front,
              float *bottom_right_front,
              float *top_left_back,
              float *top_right_back,
              float *bottom_left_back,
              float *bottom_right_back,
              float *out,
              int block_size,
              int x_in_size,
              int y_in_size,
              int z_in_size,
              int my_x_offset,
              int my_y_offset,
              int my_z_offset,
              int my_x_boundary,
              int my_y_boundary,
              int my_z_boundary,
              int x_kernel_size,
              int y_kernel_size,
              int z_kernel_size,
              int input_channels,
              int output_channels) {

  std::cout << "block_size : " << block_size << '\n';
  std::cout << "x_in_size : " << x_in_size << '\n';
  std::cout << "y_in_size : " << y_in_size << '\n';
  std::cout << "z_in_size : " << z_in_size << '\n';
  std::cout << "my_x_offset : " << my_x_offset << '\n';
  std::cout << "my_y_offset : " << my_y_offset << '\n';
  std::cout << "my_z_offset : " << my_z_offset << '\n';
  std::cout << "my_x_boundary : " << my_x_boundary << '\n';
  std::cout << "my_y_boundary : " << my_y_boundary << '\n';
  std::cout << "my_z_boundary : " << my_z_boundary << '\n';
  std::cout << "x_kernel_size : " << x_kernel_size << '\n';
  std::cout << "y_kernel_size : " << y_kernel_size << '\n';
  std::cout << "z_kernel_size : " << z_kernel_size << '\n';
  std::cout << "input_channels : " << input_channels << '\n';
  std::cout << "output_channels : " << output_channels << '\n';

  int x_size = block_size;
  int y_size = block_size;

  /// 1. Form the input tensor

  // allocate the memory
  auto *in_tmp = (float *) calloc(x_in_size * y_in_size * z_in_size * input_channels, sizeof(float));

  // -------------------------------------------------TOP-------------------------------------------------------------

  write_to_tensor(in_tmp,
                  x_in_size, y_in_size, z_in_size,
                  &get_value(top_left_front, my_x_offset, my_y_offset, my_z_offset),
                  0, 0, 0,
                  block_size - my_x_offset, block_size - my_y_offset, block_size - my_z_offset, input_channels);

  write_to_tensor(in_tmp,
                  x_in_size, y_in_size, z_in_size,
                  &get_value(top_right_front, my_x_offset, 0, my_z_offset),
                  0, block_size - my_y_offset, 0,
                  block_size - my_x_offset, block_size - my_y_boundary, block_size - my_z_offset, input_channels);

  write_to_tensor(in_tmp,
                  x_in_size, y_in_size, z_in_size,
                  &get_value(top_left_back, my_x_offset, my_y_offset, 0),
                  0, 0, block_size - my_z_offset,
                  block_size - my_x_offset, block_size - my_y_offset, block_size - my_z_boundary, input_channels);

  write_to_tensor(in_tmp,
                  x_in_size, y_in_size, z_in_size,
                  &get_value(top_right_back, my_x_offset, 0, 0),
                  0, block_size - my_y_offset, block_size - my_z_offset,
                  block_size - my_x_offset, block_size - my_y_boundary, block_size - my_z_boundary, input_channels);

  // ------------------------------------------------BOTTOM-----------------------------------------------------------

  write_to_tensor(in_tmp,
                  x_in_size, y_in_size, z_in_size,
                  &get_value(bottom_left_front, 0, my_y_offset, my_z_offset),
                  block_size - my_x_offset, 0, 0,
                  block_size - my_x_boundary, block_size - my_y_offset, block_size - my_z_offset, input_channels);

  write_to_tensor(in_tmp,
                  x_in_size, y_in_size, z_in_size,
                  &get_value(bottom_right_front, 0, 0, my_z_offset),
                  block_size - my_x_offset, block_size - my_y_offset, 0,
                  block_size - my_x_boundary, block_size - my_y_boundary, block_size - my_z_offset, input_channels);

  write_to_tensor(in_tmp,
                  x_in_size, y_in_size, z_in_size,
                  &get_value(bottom_left_back, 0, my_y_offset, 0),
                  block_size - my_x_offset, 0, block_size - my_z_offset,
                  block_size - my_x_boundary, block_size - my_y_offset, block_size - my_z_boundary, input_channels);

  write_to_tensor(in_tmp,
                  x_in_size, y_in_size, z_in_size,
                  &get_value(bottom_right_back, my_x_offset, 0, 0),
                  block_size - my_x_offset, block_size - my_y_offset, block_size - my_z_offset,
                  block_size - my_x_boundary, block_size - my_y_boundary, block_size - my_z_boundary, input_channels);

  //                            batch_size, input_channel, z, y, x
  at::Tensor a = at::from_blob(in_tmp, {1, input_channels, z_in_size, y_in_size, x_in_size});

  /// 2. do the convolution

  // make an output
  at::Tensor b = at::rand({output_channels, input_channels, 3, 3, 3});
  auto c = at::conv3d(a, b);

  std::cout << a << "\n";

  /// 3. Make the output
  // do the copy
  memcpy(out, in_tmp,input_channels * (x_in_size - x_kernel_size + 1) * (y_in_size - y_kernel_size + 1) *(z_in_size - z_kernel_size + 1));
}

}

extern "C" void process(float *top_left_front,
                        float *top_right_front,
                        float *bottom_left_front,
                        float *bottom_right_front,
                        float *top_left_back,
                        float *top_right_back,
                        float *bottom_left_back,
                        float *bottom_right_back,
                        float *out,
                        int block_size,
                        int x_out_size,
                        int y_out_size,
                        int z_out_size,
                        int my_x_offset,
                        int my_y_offset,
                        int my_z_offset,
                        int my_x_boundary,
                        int my_y_boundary,
                        int my_z_boundary,
                        int x_kernel_size,
                        int y_kernel_size,
                        int z_kernel_size,
                        int input_channels,
                        int output_channels) {
  pdb::_process(top_left_front,
                top_right_front,
                bottom_left_front,
                bottom_right_front,
                top_left_back,
                top_right_back,
                bottom_left_back,
                bottom_right_back,
                out,
                block_size,
                x_out_size,
                y_out_size,
                z_out_size,
                my_x_offset,
                my_y_offset,
                my_z_offset,
                my_x_boundary,
                my_y_boundary,
                my_z_boundary,
                x_kernel_size,
                y_kernel_size,
                z_kernel_size,
                input_channels,
                output_channels);
}


#undef get_value
#undef get_out
#undef get_conv_value