#pragma once

#include <cstdint>
#include <tuple>
#include <cmath>

namespace pdb {

struct MM3DIdx {

  // the number of nodes
  int32_t num_nodes;

  // the number of threads per node
  int32_t num_threads;

  // the number of compute sites in the cluster
  int32_t n;

  // return side
  [[nodiscard]] int32_t get_side() const {
    return cbrt(n);
  }

  [[nodiscard]] tuple<int32_t, int32_t, int32_t> get_coords(int node, int thread) const {

    // get the side
    int32_t side = cbrt(n);

    // figure out the node and the thread
    auto itd = node * num_threads + thread;

    int z = (itd / side) / side;
    int y = (itd - z * side * side) / side;
    int x = itd - y * side - z * side * side;

    return {x, y, z};
  }

  [[nodiscard]] std::tuple<int32_t, int32_t> get(int32_t x, int32_t y, int32_t z) const {

    // get the side
    int32_t side = cbrt(n);

    // get the index
    auto itd = x + side * (y + side * z);

    // figure out the node and the thread
    auto node = itd / num_threads;
    auto thread = itd % num_threads;

    // return the node itd
    return {node, thread};
  }
};

}