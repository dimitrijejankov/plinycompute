#pragma once

extern "C" void process(float *top_left_front,
                       float *top_right_front,
                       float *bottom_left_front,
                       float *bottom_right_front,
                       float *top_left_back,
                       float *top_right_back,
                       float *bottom_left_back,
                       float *bottom_right_back,
                       float *out,
                       int block_size_x,
                       int block_size_y,
                       int block_size_z,
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
                       int output_channels);