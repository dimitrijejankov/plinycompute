#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <bitset>
#include <unordered_set>
#include <GreedyPlanner.h>

int main() {

    int32_t num_blocks = 40;
    int32_t num_lhs_records = num_blocks * num_blocks;
    int32_t num_rhs_records = num_blocks * num_blocks;
    int32_t num_nodes = 40;
    int32_t num_agg_groups = num_blocks * num_blocks;

    std::vector<char> lhs_record_positions;
    lhs_record_positions.resize(num_lhs_records * num_nodes);
    for (int i = 0; i < num_lhs_records; ++i) {
        lhs_record_positions[i * num_nodes + (i % num_nodes)] = true;
    }

    std::vector<char> rhs_record_positions;
    rhs_record_positions.resize(num_rhs_records * num_nodes);
    for (int i = 0; i < num_rhs_records; ++i) {
        rhs_record_positions[i * num_nodes + (i % num_nodes)] = true;
    }

    std::vector<std::pair<int32_t, int32_t>> join_groups;
    join_groups.reserve(num_blocks * num_blocks * num_blocks);

    for (int i = 0; i < num_blocks; ++i) {
        for (int j = 0; j < num_blocks; ++j) {
            for (int k = 0; k < num_blocks; ++k) {

                int l = i * num_blocks + k;
                int r = k * num_blocks + j;
                join_groups.emplace_back(l, r);
            }
        }
    }

    std::vector<std::vector<int32_t>> aggregation_groups;
    aggregation_groups.resize(num_agg_groups);

    for (int i = 0; i < num_agg_groups; ++i) {
        aggregation_groups[i].resize(num_blocks);
        int start = i * num_blocks;
        for (int j = 0; j < num_blocks; ++j) {
            aggregation_groups[i][j] = start + j;
        }
    }


    pdb::GreedyPlanner::costs_t c{};
    c.agg_cost = 1;
    c.join_cost = 1;
    c.join_rec_size = 1;
    c.send_coef = 1;
    c.rhs_input_rec_size = 1;
    c.lhs_input_rec_size = 1;
    c.aggregation_rec_size = 1;

    pdb::GreedyPlanner planner(num_nodes,
                               c,
                               lhs_record_positions,
                               rhs_record_positions,
                               aggregation_groups,
                               join_groups);

    planner.run_join_first_only();

    planner.print();
    return 0;
}
