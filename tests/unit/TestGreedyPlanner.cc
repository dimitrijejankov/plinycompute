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

class TestGreedyPlanner {
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
        std::vector<int32_t> join_costs ;

        std::vector<int32_t> join_assignments ;

        std::vector<int32_t> shuffle_costs;

        int32_t join_shuffle;

        // the node where to put this
        int32_t agg_node = 0;

        // figure out the aggregation cost
        int32_t aggregation_costs;

        // the total overhead
        int32_t total_cost = 0;
    };

    TestGreedyPlanner(int32_t numNodes,
                      costs_t costs,
                      std::vector<char> lhsRecordPositions,
                      std::vector<char> rhsRecordPositions,
                      std::vector<std::vector<int32_t>> aggregationGroups,
                      std::vector<std::pair<int32_t, int32_t>> joinGroups)
            : num_nodes(numNodes),
              num_agg_groups(aggregationGroups.size()),
              num_join_groups(joinGroups.size()),
              num_lhs_records(lhsRecordPositions.size()),
              num_rhs_records(rhsRecordPositions.size()),
              c(costs),
              lhs_record_positions(std::move(lhsRecordPositions)),
              rhs_record_positions(std::move(rhsRecordPositions)),
              aggregation_groups(std::move(aggregationGroups)),
              join_groups(std::move(joinGroups)) {


        max_join_projection_cost = 0;
        max_side_shuffling_cost = 0;
        max_agg_projection_cost = 0;
        max_join_shuffling_cost = 0;

        join_projection_costs.resize(numNodes);
        side_shuffling_costs.resize(numNodes);
        agg_projection_costs.resize(numNodes);
        join_side_shuffling_costs.resize(numNodes);

        join_group_positions.resize(num_join_groups * numNodes);
        aggregation_positions.resize(num_agg_groups * numNodes);

        tmpLHSFetches.resize(numNodes);
        tmpRHSFetches.resize(numNodes);
    }


    void run() {

        // init the aggregation groups
        std::vector<int32_t> aggGroups;
        aggGroups.resize(num_agg_groups);
        for (int i = 0; i < num_agg_groups; ++i) {
            aggGroups[i] = i;
        }

        // get a permutation
        std::random_device rd;
        std::mt19937 gen(rd());
        //std::shuffle(aggGroups.begin(), aggGroups.end(), gen);

        // go through the aggregation groups
        for (auto g : aggGroups) {

            // get the join groups
            auto &joinGroups = aggregation_groups[g];

            // the the aggregation choice
            auto agg_choice = try_assign_agg_group(joinGroups);

            // the join choice
            auto join_choice = try_assign_join_group(joinGroups);

            // figure out which one is better
            if(agg_choice.total_cost <= join_choice.total_cost) {
                apply_agg(agg_choice, joinGroups, g);
            }
            else {
                apply_join(join_choice, joinGroups, g);
            }
        }

    }

    agg_plan_t try_assign_agg_group(std::vector<int32_t> &joinGroups) {

        // go through all nodes and try to figure out here to place the aggregation group
        agg_plan_t min_node{};
        for (int node = 0; node < num_nodes; ++node) {

            // figure out the aggregation cost
            auto aggregation_cost = agg_projection_costs[node] + c.agg_cost;

            // figure out the join cost
            auto join_cost = join_projection_costs[node] + c.join_cost * joinGroups.size();

            // figure out shuffle cost for each join
            lhs_tmp.clear();
            rhs_tmp.clear();
            for (auto jg : joinGroups) {

                // get the joined lhs and rhs
                auto &p = join_groups[jg];

                // ok now tha te go them assign them to this node
                lhs_tmp.insert(p.first);
                rhs_tmp.insert(p.second);
            }

            // sum the costs for the left side
            int32_t shuffle_cost = 0;
            for (auto &l : lhs_tmp) {

                // if it is not assigned we need to fetch it
                if (lhs_record_positions[l * num_nodes + node] == 0) {
                    shuffle_cost += c.lhs_input_rec_size;
                }
            }

            // sum the cost for the right side
            for (auto &r : rhs_tmp) {

                // if it is not assigned we need to fetch it
                if (rhs_record_positions[r * num_nodes + node] == 0) {
                    shuffle_cost += c.rhs_input_rec_size;
                }
            }

            // add the current cost for this node
            shuffle_cost *= c.send_coef;
            shuffle_cost += side_shuffling_costs[node];

            // figure out the cost
            auto overhead_agg = max_agg_projection_cost >= aggregation_cost ? 0 : aggregation_cost - agg_projection_costs[node];
            auto overhead_join = max_join_projection_cost >= join_cost ? 0 : join_cost - join_projection_costs[node];
            auto overhead_shuffle = max_side_shuffling_cost >= shuffle_cost ? 0 : shuffle_cost - side_shuffling_costs[node];
            auto cost = overhead_agg + overhead_join + overhead_shuffle;

            // set the cost
            if(min_node.total_cost > cost) {
                min_node.node = node;
                min_node.total_cost = cost;
                min_node.join_cost = join_cost;
                min_node.aggregation_cost = aggregation_cost;
                min_node.shuffle_cost = shuffle_cost;
            }
        }

        // return the min node
        return min_node;
    }

    void apply_agg(const agg_plan_t &pl, std::vector<int32_t> &joinGroups, int32_t aggGroup) {


        // figure out the aggregation cost
        agg_projection_costs[pl.node] = pl.aggregation_cost;
        max_agg_projection_cost = std::max(max_agg_projection_cost, pl.aggregation_cost);

        // figure out the join cost
        join_projection_costs[pl.node] = pl.join_cost;
        max_join_projection_cost = std::max(max_join_projection_cost, pl.join_cost);

        // figure out shuffle cost
        side_shuffling_costs[pl.node] = pl.shuffle_cost;
        max_side_shuffling_cost = std::max(max_side_shuffling_cost, pl.shuffle_cost);

        // update the lhs and rhs positions
        for(auto &jg : joinGroups) {

            // get the joined lhs and rhs
            auto &p = join_groups[jg];

            // mark that we have
            lhs_record_positions[p.first * num_nodes + pl.node] = true;
            rhs_record_positions[p.first * num_nodes + pl.node] = true;

            // assign join group
            join_group_positions[jg * num_nodes + pl.node] = true;
        }

        // set the aggregation group position
        aggregation_positions[aggGroup * num_nodes + pl.node] = true;
    }

    join_plan_t try_assign_join_group(std::vector<int32_t> &joinGroups) {

        // total ovehead
        int32_t total_overhead = 0;

        // copy the shuffling costs
        auto planned_shuffle_cost = side_shuffling_costs;
        auto planned_max_shuffle_cost = max_side_shuffling_cost;

        // copy the join costs
        auto planned_join_costs = join_projection_costs;
        auto planned_max_join_cost = max_join_projection_cost;

        // this keeps track of where we assigned the join groups
        planned_join_assignments.clear();

        // go and assign each join group
        for(auto jg : joinGroups) {

            // get the joined lhs and rhs
            auto &p = join_groups[jg];

            // the best node and the best cost so far
            int32_t best_node = 0;
            int32_t best_cost = std::numeric_limits<int32_t>::max();

            // the best costs for lhs and rhs
            int32_t best_shuffle_cost = 0;
            int32_t best_join_cost = 0;

            // go through the nodes
            for (int node = 0; node < num_nodes; ++node) {

                int32_t lhs_cost = 0;
                int32_t rhs_cost = 0;

                // is the rhs of the join group off this node
                auto lhsOffNode = lhs_record_positions[p.first * num_nodes + node] == 0 &&
                                  std::find(tmpLHSFetches[node].begin(), tmpLHSFetches[node].end(), p.first) == tmpLHSFetches[node].end();

                // is the lhs of the join group off this node
                auto rhsOffNode = rhs_record_positions[p.second * num_nodes + node] == 0 &&
                                  std::find(tmpRHSFetches[node].begin(), tmpRHSFetches[node].end(), p.second) == tmpRHSFetches[node].end();

                // if the lhs is of the node mark that we fetched it and update the cost
                if(lhsOffNode) {
                    tmpLHSFetches[node].emplace_back(p.first);
                    lhs_cost = c.send_coef * c.lhs_input_rec_size;
                }

                // if the rhs if off the node mark that we have fetched and update the cost
                if(rhsOffNode) {
                    tmpRHSFetches[node].emplace_back(p.second);
                    rhs_cost = c.send_coef * c.rhs_input_rec_size;
                }

                // this is how much the total shuffling cost is
                auto shuffle_cost = lhs_cost + rhs_cost + planned_shuffle_cost[node];
                auto join_cost = lhs_cost + rhs_cost + planned_shuffle_cost[node];

                // figure out how much overhead we are adding
                auto overhead_join = planned_max_join_cost >= join_cost ? 0 : join_cost - planned_join_costs[node];
                auto overhead_shuffle = planned_max_shuffle_cost >= shuffle_cost ? 0 : shuffle_cost - planned_shuffle_cost[node];

                // the current cost is the overhead of fetching the sides and performing the join projection
                auto currentCost = overhead_join + overhead_shuffle;

                // do we have a better option if so save it
                if(currentCost < best_cost) {
                    best_node = node;
                    best_cost = currentCost;
                    best_shuffle_cost = shuffle_cost;
                    best_join_cost = join_cost;
                }
            }

            // update the overhead
            total_overhead += best_cost;

            // update the planned costs
            planned_shuffle_cost[best_node] = best_shuffle_cost;
            planned_max_shuffle_cost = std::max(best_shuffle_cost, planned_max_shuffle_cost);
            planned_join_costs[best_node] = best_join_cost;
            planned_max_join_cost = std::max(planned_max_join_cost, best_join_cost);

            // store the planned
            planned_join_assignments.emplace_back(best_node);
        }

        // do the aggregation group
        int32_t best_agg_node = 0;
        int32_t best_cost = std::numeric_limits<int32_t>::max();
        int32_t best_agg_projection = std::numeric_limits<int32_t>::max();
        int32_t best_join_shuffle = std::numeric_limits<int32_t>::max();

        // go through the nodes and assign
        for (int node = 0; node < num_nodes; ++node) {

            // go and see what we need to fetch
            int32_t current_shuffle_cost = join_side_shuffling_costs[node];
            for(int i = 0; i < joinGroups.size(); ++i) {
                if(planned_join_assignments[i] != node || join_group_positions[joinGroups[i] * num_nodes + node] == 0) {
                    current_shuffle_cost += c.send_coef * c.join_rec_size;
                }
            }

            // figure out what the overhead is
            auto overhead_shuffle = max_join_shuffling_cost >= current_shuffle_cost ? 0 : current_shuffle_cost - agg_projection_costs[node];

            // if we made an unbalance add cost
            int32_t current_projection_cost = agg_projection_costs[node] + c.agg_cost;
            int32_t overhead_projection_cost = max_agg_projection_cost >= current_projection_cost ? 0 : current_projection_cost - agg_projection_costs[node];

            // figure out the total overhead cost
            int32_t current_cost = overhead_projection_cost + overhead_shuffle;

            // save the node if better
            if(current_cost < best_cost) {
                best_cost = current_cost;
                best_agg_projection = current_projection_cost;
                best_join_shuffle = current_shuffle_cost;
                best_agg_node = node;
            }
        }

        // update the overhead
        total_overhead += best_cost;

        // fill out the return value
        join_plan_t out;
        out.shuffle_costs = std::move(planned_shuffle_cost);
        out.join_assignments = planned_join_assignments;
        out.join_costs = std::move(planned_join_costs);
        out.agg_node = best_agg_node;
        out.aggregation_costs = best_agg_projection;
        out.join_shuffle = best_join_shuffle;
        out.total_cost = total_overhead;

        // make a tuple contain
        return std::move(out);
    }

    void apply_join(const join_plan_t &pl, std::vector<int32_t> &joinGroups, int32_t aggGroup) {

        // go through all the join groups
        for(int i = 0; i < joinGroups.size(); ++i) {

            // get the joined lhs and rhs
            auto &p = join_groups[joinGroups[i]];

            // set the assignments
            lhs_record_positions[p.first * num_nodes + pl.join_assignments[i]] = true;
            rhs_record_positions[p.second * num_nodes + pl.join_assignments[i]] = true;

            // assign join group
            join_group_positions[i * num_nodes + pl.join_assignments[i]] = true;
        }

        // set the assignment
        aggregation_positions[aggGroup * num_nodes + pl.agg_node] = true;

        // go through all the nodes and update the costs for the join
        for(int node = 0; node < num_nodes; ++node) {

            // update the join costs
            side_shuffling_costs[node] = pl.shuffle_costs[node];
            join_projection_costs[node] = pl.join_costs[node];

            // update the max shuffling costs if necessary
            max_side_shuffling_cost = std::max(max_side_shuffling_cost, side_shuffling_costs[node]);
            max_join_projection_cost = std::max(max_join_projection_cost, join_projection_costs[node]);
        }

        // update the max shuffling costs
        max_join_shuffling_cost = std::max(max_join_shuffling_cost, pl.join_shuffle);
        max_agg_projection_cost = std::max(max_agg_projection_cost, pl.aggregation_costs);

        join_side_shuffling_costs[pl.agg_node] = pl.join_shuffle;
        agg_projection_costs[pl.agg_node] = pl.aggregation_costs;
    }

    std::unordered_set<int32_t> lhs_tmp;
    std::unordered_set<int32_t> rhs_tmp;

    std::vector<int32_t> planned_join_assignments;
    std::vector<std::vector<int32_t>> tmpLHSFetches;
    std::vector<std::vector<int32_t>> tmpRHSFetches;

    int32_t max_side_shuffling_cost;
    std::vector<int32_t> side_shuffling_costs;

    int32_t max_join_projection_cost;
    std::vector<int32_t> join_projection_costs;

    int32_t max_join_shuffling_cost;
    std::vector<int32_t> join_side_shuffling_costs;

    int32_t max_agg_projection_cost;
    std::vector<int32_t> agg_projection_costs;

    int32_t num_nodes = 3;
    int32_t num_agg_groups = 16;
    int32_t num_join_groups = 16;
    int32_t num_lhs_records;
    int32_t num_rhs_records;

    // [node][join_group]
    std::vector<char> lhs_record_positions;

    // [node][join_group]
    std::vector<char> rhs_record_positions;

    std::vector<int32_t> aggregation_positions;

    std::vector<int32_t> join_group_positions;

    std::vector<std::vector<int32_t>> aggregation_groups;

    std::vector<std::pair<int32_t, int32_t>> join_groups;

    costs_t c;
};

int main() {

    int32_t num_blocks = 4;
    int32_t num_lhs_records = num_blocks * num_blocks;
    int32_t num_rhs_records = num_blocks * num_blocks;
    int32_t num_nodes = 4;
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


    TestGreedyPlanner::costs_t c{};
    c.agg_cost = 1;
    c.join_cost = 1;
    c.join_rec_size = 1;
    c.send_coef = 1;
    c.rhs_input_rec_size = 1;
    c.lhs_input_rec_size = 1;
    c.aggregation_rec_size = 1;

    TestGreedyPlanner planner(num_nodes,
                              c,
                              lhs_record_positions,
                              rhs_record_positions,
                              aggregation_groups,
                              join_groups);

    planner.run();

    return 0;
}
