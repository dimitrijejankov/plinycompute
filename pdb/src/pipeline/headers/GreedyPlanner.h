#pragma once

#include <utility>
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <iostream>

namespace pdb {

class GreedyPlanner {
 public:

  // these are the costs_t we have for the genetic algorithm
  struct costs_t {

    // how large is the lhs input record
    int32_t lhs_input_rec_size;

    // how large is the rhs input record
    int32_t rhs_input_rec_size;

    // how large is the join record
    int32_t join_rec_size;

    // how large is the aggregated record
    int32_t aggregation_rec_size;

    // sending is modeled as send_coef * size + const, we ignore the constant since we assume it
    // to be small compared to the size of the page
    int32_t send_coef;

    // join projection cost is the cost to run the pipeline for the join to the part where we need to aggregate
    int32_t join_cost;

    // the cost to perform the aggregation
    int32_t agg_cost;
  };

  struct agg_plan_t {

    // figure out the aggregation cost
    int32_t aggregation_cost = 0;

    // figure out the join cost
    int32_t join_cost = 0;

    int32_t shuffle_cost = 0;

    int32_t node = 0;

    int32_t total_cost = std::numeric_limits<int32_t>::max();
  };

  struct join_plan_t {

    // figure out the join cost
    std::vector<int32_t> join_costs;

    std::vector<int32_t> join_assignments;

    std::vector<int32_t> shuffle_costs;

    int32_t join_shuffle{};

    // the node where to put this
    int32_t agg_node = 0;

    // figure out the aggregation cost
    int32_t aggregation_costs{};

    // the total overhead
    int32_t total_cost = 0;
  };

  struct planning_result {

    planning_result(std::vector<int32_t> agg_group_assignments,
                    std::vector<int32_t> join_groups_node)
        : agg_group_assignments(std::move(agg_group_assignments)),
          join_groups_to_node(std::move(join_groups_node)) {}

    // tells us what node the aggregation group was assigned
    std::vector<int32_t> agg_group_assignments;

    // tells us
    std::vector<int32_t> join_groups_to_node;
  };

  GreedyPlanner(int32_t numNodes, pdb::GreedyPlanner::costs_t costs,
                const std::vector<char> &lhsRecordPositions, const std::vector<char> &rhsRecordPositions,
                const std::vector<std::vector<int32_t>> &aggregationGroups,
                const std::vector<std::pair<int32_t, int32_t>> &joinGroups);

  void run();

  void run_agg_first_only();

  void run_join_first_only();

  std::vector<int32_t> get_agg_result();

  planning_result get_result();

  [[nodiscard]] int32_t getCost() const;

  agg_plan_t try_assign_agg_group(std::vector<int32_t> &joinGroups);

  void apply_agg(const agg_plan_t &pl, std::vector<int32_t> &joinGroups, int32_t aggGroup);

  join_plan_t try_assign_join_group(std::vector<int32_t> &joinGroups);

  void apply_join(const join_plan_t &pl, std::vector<int32_t> &joinGroups, int32_t aggGroup);

  void print();

  // this is used when we plan try to plan by fitting the join groups first
  std::vector<int32_t> planned_shuffle_cost;
  std::vector<int32_t> planned_join_costs;
  std::vector<int32_t> planned_join_assignments;

  // this keeps track of the changes that we need to revert
  std::vector<std::vector<int32_t>> tmpLHSFetches;
  std::vector<std::vector<int32_t>> tmpRHSFetches;

  // this is used when we plan the aggregation group only assignment to figure out the required records quickly
  std::vector<int32_t> lhs_tmp;
  std::vector<int32_t> lhs_tmp_rst;
  std::vector<int32_t> rhs_tmp;
  std::vector<int32_t> rhs_tmp_rst;

  // these are the costs of shuffling the lhs and rhs records per node
  int32_t max_side_shuffling_cost;
  std::vector<int32_t> side_shuffling_costs;

  // these are the costs of performing the join projection per node
  int32_t max_join_projection_cost;
  std::vector<int32_t> join_projection_costs;

  // these are the shuffling costs for the join groups per node
  int32_t max_join_shuffling_cost;
  std::vector<int32_t> join_side_shuffling_costs;

  // these are the costs for the aggregation projection per node
  int32_t max_agg_projection_cost;
  std::vector<int32_t> agg_projection_costs;

  // these is the info about the planning
  int32_t num_nodes = 3;
  int32_t num_agg_groups = 16;
  int32_t num_join_groups = 16;
  int32_t num_lhs_records;
  int32_t num_rhs_records;

  // keeps track of where the lhs records need to be sent [lhs][node]
  std::vector<char> lhs_record_positions;

  // keeps track of where the rhs records need to be sent [rhs][node]
  std::vector<char> rhs_record_positions;

  // keeps track of where the aggregation groups are assigned [agg][node]
  std::vector<int32_t> aggregation_positions;

  // keeps track of where the join groups are assigned [jg][node]
  std::vector<int32_t> join_group_positions;

  // tells us what join group each aggregation group consists of
  std::vector<std::vector<int32_t>> aggregation_groups;

  // tells us what lhs and rhs record each join group consists of
  std::vector<std::pair<int32_t, int32_t>> join_groups;

  // the costs of performing each stuff
  costs_t c;
};

}
