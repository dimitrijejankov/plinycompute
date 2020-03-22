#pragma once

#include <random>

namespace pdb {

struct SingleNodePlanner {

  int32_t num_agg_groups;
  std::vector<int32_t> result;

  explicit SingleNodePlanner(int32_t num_agg_groups) : num_agg_groups(num_agg_groups) {
    result.resize(num_agg_groups);
  }

  auto &getResult() {
    return result;
  }

  void print() {

    for (int gene : result) {
      std::cout << gene << "| ";
    }

    std::cout << '\n';
  }

};

}