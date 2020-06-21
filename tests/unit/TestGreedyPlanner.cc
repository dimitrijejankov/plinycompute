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
#include <PipJoinAggPlanResult.h>

struct record_t {

  int32_t rowID;
  int32_t colID;
};

bool operator==(const record_t &lhs, const record_t &rhs) {
  return lhs.rowID == rhs.rowID && lhs.colID == rhs.colID;
}

struct record_hasher_t {
  std::size_t operator()(record_t const &s) const noexcept {
    std::size_t h1 = std::hash<int32_t>{}(s.rowID);
    std::size_t h2 = std::hash<int32_t>{}(s.colID);
    return h1 ^ (h2 << 1);
  }
};

struct join_group_t {
  int32_t lhs;
  int32_t rhs;
  int32_t agg = -1;
};

bool operator==(const join_group_t &lhs, const join_group_t &rhs) {
  return lhs.lhs == rhs.lhs && lhs.rhs == rhs.rhs;
}

struct join_group_hasher_t {
  std::size_t operator()(join_group_t const &s) const noexcept {
    std::size_t h1 = std::hash<int32_t>{}(s.lhs);
    std::size_t h2 = std::hash<int32_t>{}(s.rhs);
    return h1 ^ (h2 << 1);
  }
};

int main() {

  uint64_t num_row_ids_a = 80;
  uint64_t num_col_ids_a = 80;
  uint64_t num_row_ids_b = 80;
  uint64_t num_col_ids_b = 80;
  uint64_t num_nodes = 10;

  // the lhs records
  std::vector<record_t> lhs_records;
  for (auto row_id_a = 0; row_id_a < num_row_ids_a; ++row_id_a) {
    for (auto col_id_a = 0; col_id_a < num_col_ids_a; ++col_id_a) {
      lhs_records.push_back({.rowID = row_id_a, .colID = col_id_a});
    }
  }

  // the rhs records
  std::vector<record_t> rhs_records;
  for (auto row_id_b = 0; row_id_b < num_row_ids_b; ++row_id_b) {
    for (auto col_id_b = 0; col_id_b < num_col_ids_b; ++col_id_b) {
      rhs_records.push_back({.rowID = row_id_b, .colID = col_id_b});
    }
  }

  // join groups
  std::unordered_map<join_group_t, int32_t, join_group_hasher_t> join_groups;
  std::unordered_map<record_t, std::vector<int32_t>, record_hasher_t> agg_groups;
  for (auto lhs = 0; lhs < lhs_records.size(); ++lhs) {
    for (auto rhs = 0; rhs < rhs_records.size(); ++rhs) {

      // check if they match
      if (lhs_records[lhs].colID == rhs_records[rhs].rowID) {

        // make the key
        join_group_t key{.lhs = lhs, .rhs = rhs};

        // try to find the
        auto it = join_groups.find(key);
        int32_t id = 0;

        // try to find it, if not assign it
        if (it == join_groups.end()) { id = (join_groups[key] = join_groups.size()); } else { id = it->second; }

        // add it to the aggregation group
        agg_groups[record_t{.rowID = lhs_records[lhs].rowID, .colID = rhs_records[rhs].colID}].push_back(id);
      }
    }
  }

  // set the costs
  pdb::GreedyPlanner::costs_t c{};
  c.agg_cost = 1;
  c.join_cost = 1;
  c.join_rec_size = 1;
  c.send_coef = 1;
  c.rhs_input_rec_size = 1;
  c.lhs_input_rec_size = 1;
  c.aggregation_rec_size = 1;

  // make the lhs record positions
  std::vector<char> lhs_record_positions;
  lhs_record_positions.resize(lhs_records.size() * num_nodes);
  for (int i = 0; i < lhs_records.size(); ++i) {
    lhs_record_positions[i * num_nodes + (i % num_nodes)] = true;
  }

  // make the rhs record positions
  std::vector<char> rhs_record_positions;
  rhs_record_positions.resize(rhs_records.size() * num_nodes);
  for (int i = 0; i < rhs_records.size(); ++i) {
    rhs_record_positions[i * num_nodes + (i % num_nodes)] = true;
  }

  //
  std::vector<pdb::PipJoinAggPlanResult::JoinedRecord> join_groups_vec;
  join_groups_vec.resize(join_groups.size());
  for (auto &jg : join_groups) {
    join_groups_vec[jg.second] = pdb::PipJoinAggPlanResult::JoinedRecord{ static_cast<uint32_t>(jg.first.lhs),
                                                                          static_cast<uint32_t>(jg.first.rhs),
                                                                          0u};
  }

  std::vector<std::vector<int32_t>> aggregation_groups;
  aggregation_groups.resize(agg_groups.size());
  int32_t i = 0;
  for (auto &agg : agg_groups) {
    for (auto &jg : agg.second) {
      aggregation_groups[i].emplace_back(jg);
      join_groups_vec[jg].aggTID = i;
    }
  }

  std::cout << "Join groups size " << join_groups_vec.size() << "\n";
  std::cout << "agg groups size " << aggregation_groups.size() << "\n";
  pdb::GreedyPlanner planner(num_nodes,
                             c,
                             lhs_record_positions,
                             rhs_record_positions,
                             aggregation_groups,
                             join_groups_vec);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  planner.run();
  planner.print();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Planner run for " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "[ns]" << '\n';

  planner.print();
  return 0;
}
