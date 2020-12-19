#pragma once

#include <utility>
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <iostream>
#include <PipJoinAggPlanResult.h>
#include <pipeline/Join3KeyPipeline.h>

namespace pdb {

class GreedyPlanner {
 public:

  // these are the costs_t we have for the genetic algorithm
  struct costs_t {

    // how large is the lhs input record
    int32_t rec_size;

    // sending is modeled as send_coef * size + const, we ignore the constant since we assume it
    // to be small compared to the size of the page
    int32_t send_coef;

    // join projection cost is the cost to run the pipeline for the join to the part where we need to aggregate
    int32_t join_cost;
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

  GreedyPlanner(int32_t numNodes,
                pdb::GreedyPlanner::costs_t costs,
                const std::vector<char> &side_record_positions,
                const std::vector<Join3KeyPipeline::joined_record> &joinedRecords);

  void run_join_first_only();

  std::vector<int32_t> get_result();

  void print();

  // this is used when we plan try to plan by fitting the join groups first
  int32_t planned_max_shuffle_cost;
  std::vector<int32_t> planned_shuffle_cost;

  // the max planned join projection cost
  int32_t planned_max_join_cost;
  std::vector<int32_t> planned_join_costs;

  // the total overhead of the plan
  int32_t total_overhead;

  // these is the info about the planning
  int32_t num_nodes = 3;
  int32_t num_join_groups = 16;
  int32_t num_join_records;

  // keeps track of where the lhs records need to be sent [lhs][node]
  std::vector<char> side_record_positions;

  // keeps track of where the join groups are assigned [jg][node]
  std::vector<int32_t> join_group_positions;

  // tells us what lhs record each join group consists of
  const std::vector<Join3KeyPipeline::joined_record> &join_groups;

  // the costs of performing each stuff
  costs_t c;
};

}
