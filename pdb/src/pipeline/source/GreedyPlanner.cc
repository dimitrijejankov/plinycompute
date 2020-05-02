#include <GreedyPlanner.h>

pdb::GreedyPlanner::GreedyPlanner(int32_t numNodes, pdb::GreedyPlanner::costs_t costs,
                                  const std::vector<char> &side_record_positions,
                                  const std::vector<EightWayJoinPipeline::joined_record> &joinedRecords)
    : num_nodes(numNodes),
      num_join_records(side_record_positions.size() / num_nodes),
      c(costs),
      side_record_positions(side_record_positions),
      join_groups(joinedRecords) {

  planned_max_shuffle_cost = 0;
  planned_max_join_cost = 0;
  total_overhead = 0;

  join_group_positions.resize(joinedRecords.size());
  planned_shuffle_cost.resize(numNodes);
  planned_join_costs.resize(numNodes);
}
void pdb::GreedyPlanner::run_join_first_only() {

  // go through the aggregation groups
  for (int i = 0; i < join_groups.size(); ++i) {

    // get the join group
    auto &p = join_groups[i];

    // the best node and the best cost so far
    int32_t best_node = 0;
    int32_t best_cost = std::numeric_limits<int32_t>::max();

    // the best costs for lhs and rhs
    int32_t best_shuffle_cost = 0;
    int32_t best_join_cost = 0;

    // go through the nodes find the best node to assign the join group to
    for (int node = 0; node < num_nodes; ++node) {

      int32_t record_cost = 0;

      // is the rhs of the join group off this node
      auto firstOffNode = side_record_positions[p.first * num_nodes + node] == 0;
      auto secondOffNode = side_record_positions[p.second * num_nodes + node] == 0;
      auto thirdOffNode = side_record_positions[p.third * num_nodes + node] == 0;
      auto fourthOffNode = side_record_positions[p.fourth * num_nodes + node] == 0;
      auto fifthOffNode = side_record_positions[p.fifth * num_nodes + node] == 0;
      auto sixthOffNode = side_record_positions[p.sixth * num_nodes + node] == 0;
      auto seventhOffNode = side_record_positions[p.seventh * num_nodes + node] == 0;
      auto eightOffNode = side_record_positions[p.eigth * num_nodes + node] == 0;

      // figure out the cost
      record_cost += (firstOffNode + secondOffNode + thirdOffNode + fourthOffNode +
          fifthOffNode + sixthOffNode + seventhOffNode + eightOffNode) * c.send_coef * c.rec_size;;

      // this is how much the total shuffling cost is
      auto shuffle_cost = record_cost + planned_shuffle_cost[node];
      auto join_cost = planned_join_costs[node] + c.join_cost;

      // figure out how much overhead we are adding
      auto overhead_join = planned_max_join_cost >= join_cost ? 0 : join_cost - planned_join_costs[node];
      auto overhead_shuffle = planned_max_shuffle_cost >= shuffle_cost ? 0 : shuffle_cost - planned_shuffle_cost[node];

      // the current cost is the overhead of fetching the sides and performing the join projection
      auto currentCost = overhead_join + overhead_shuffle;

      // do we have a better option if so save it
      if (currentCost < best_cost) {
        best_node = node;
        best_cost = currentCost;
        best_shuffle_cost = shuffle_cost;
        best_join_cost = join_cost;
      }
    }

    // store the changes for first record
    if (side_record_positions[p.first * num_nodes + best_node] == 0) {
      side_record_positions[p.first * num_nodes + best_node] = true;
    }

    // store the changes for second record
    if (side_record_positions[p.second * num_nodes + best_node] == 0) {
      side_record_positions[p.second * num_nodes + best_node] = true;
    }

    // store the changes for third record
    if (side_record_positions[p.third * num_nodes + best_node] == 0) {
      side_record_positions[p.third * num_nodes + best_node] = true;
    }

    // store the changes for fourth record
    if (side_record_positions[p.fourth * num_nodes + best_node] == 0) {
      side_record_positions[p.fourth * num_nodes + best_node] = true;
    }

    // store the changes for fifth record
    if (side_record_positions[p.fifth * num_nodes + best_node] == 0) {
      side_record_positions[p.fifth * num_nodes + best_node] = true;
    }

    // store the changes for sixth record
    if (side_record_positions[p.sixth * num_nodes + best_node] == 0) {
      side_record_positions[p.sixth * num_nodes + best_node] = true;
    }

    // store the changes for seventh record
    if (side_record_positions[p.seventh * num_nodes + best_node] == 0) {
      side_record_positions[p.seventh * num_nodes + best_node] = true;
    }

    // store the changes for eight record
    if (side_record_positions[p.eigth * num_nodes + best_node] == 0) {
      side_record_positions[p.eigth * num_nodes + best_node] = true;
    }

    // update the overhead
    total_overhead += best_cost;

    // update the planned costs
    planned_shuffle_cost[best_node] = best_shuffle_cost;
    planned_max_shuffle_cost = std::max(best_shuffle_cost, planned_max_shuffle_cost);
    planned_join_costs[best_node] = best_join_cost;
    planned_max_join_cost = std::max(planned_max_join_cost, best_join_cost);

    // set the group position
    join_group_positions[i] = best_node;
  }
}

void pdb::GreedyPlanner::print() {

  std::cout << "\n\n--join group assignments--\n";
  for (int i = 0; i < num_join_groups; ++i) {
    std::cout << "group " << i << " assigned to : \n";
    std::cout << join_group_positions[i] << " node\n";
  }
}

std::vector<int32_t> pdb::GreedyPlanner::get_result() {
  return std::move(join_group_positions);
}
