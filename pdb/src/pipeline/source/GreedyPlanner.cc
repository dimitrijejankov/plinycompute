#include <GreedyPlanner.h>

pdb::GreedyPlanner::GreedyPlanner(int32_t numNodes, pdb::GreedyPlanner::costs_t costs,
                                  std::vector<char> lhsRecordPositions, std::vector<char> rhsRecordPositions,
                                  std::vector<std::vector<int32_t>> aggregationGroups,
                                  std::vector<std::pair<int32_t, int32_t>> joinGroups)
        : num_nodes(numNodes),
          num_agg_groups(aggregationGroups.size()),
          num_join_groups(joinGroups.size()),
          num_lhs_records(lhsRecordPositions.size() / num_nodes),
          num_rhs_records(rhsRecordPositions.size() / num_nodes),
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

    lhs_tmp.resize(num_lhs_records);
    rhs_tmp.resize(num_rhs_records);
}

void pdb::GreedyPlanner::run() {

    // init the aggregation groups
    std::vector<int32_t> aggGroups;
    aggGroups.resize(num_agg_groups);
    for (int i = 0; i < num_agg_groups; ++i) {
        aggGroups[i] = i;
    }

    // get a permutation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(aggGroups.begin(), aggGroups.end(), gen);

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

void pdb::GreedyPlanner::run_agg_first_only() {

    // init the aggregation groups
    std::vector<int32_t> aggGroups;
    aggGroups.resize(num_agg_groups);
    for (int i = 0; i < num_agg_groups; ++i) {
        aggGroups[i] = i;
    }

    // get a permutation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(aggGroups.begin(), aggGroups.end(), gen);

    // go through the aggregation groups
    for (auto g : aggGroups) {

        // get the join groups
        auto &joinGroups = aggregation_groups[g];

        // the the aggregation choice
        auto agg_choice = try_assign_agg_group(joinGroups);

        // figure out which one is better
        apply_agg(agg_choice, joinGroups, g);
    }
}

void pdb::GreedyPlanner::run_join_first_only() {

    // init the aggregation groups
    std::vector<int32_t> aggGroups;
    aggGroups.resize(num_agg_groups);
    for (int i = 0; i < num_agg_groups; ++i) {
        aggGroups[i] = i;
    }

    // get a permutation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(aggGroups.begin(), aggGroups.end(), gen);

    // go through the aggregation groups
    for (auto g : aggGroups) {

        // get the join groups
        auto &joinGroups = aggregation_groups[g];

        // the join choice
        auto join_choice = try_assign_join_group(joinGroups);

        // figure out which one is better
        apply_join(join_choice, joinGroups, g);
    }
}

std::vector<int32_t> pdb::GreedyPlanner::get_agg_result() {

    std::vector<int32_t> result;
    result.resize(num_agg_groups);

    // go through the assignments
    for(int i = 0; i < num_agg_groups; ++i) {
        for(int j = 0; j < num_nodes; j++) {
            if(aggregation_positions[i * num_nodes + j]) {
                result[i] = j;
            }
        }
    }

    return std::move(result);
}

pdb::GreedyPlanner::agg_plan_t pdb::GreedyPlanner::try_assign_agg_group(std::vector<int32_t> &joinGroups) {

    // go through all nodes and try to figure out here to place the aggregation group
    agg_plan_t min_node{};
    for (int node = 0; node < num_nodes; ++node) {

        // figure out the aggregation cost
        auto aggregation_cost = agg_projection_costs[node] + c.agg_cost;

        // figure out the join cost
        auto join_cost = join_projection_costs[node] + c.join_cost * joinGroups.size();

        // figure out shuffle cost for each join
        lhs_tmp_rst.clear();
        rhs_tmp_rst.clear();
        for (auto jg : joinGroups) {

            // get the joined lhs and rhs
            auto &p = join_groups[jg];

            // ok now tha te go them assign them to this node
            lhs_tmp[p.first] = 1;
            lhs_tmp_rst.emplace_back(p.first);
            rhs_tmp[p.second] = 1;
            rhs_tmp_rst.emplace_back(p.second);
        }

        // sum the costs for the left side
        int32_t shuffle_cost = 0;
        for (auto l = 0; l < lhs_tmp.size(); l++) {

            // if it is not assigned we need to fetch it
            if (lhs_record_positions[l * num_nodes + node] == 0 && lhs_tmp[l] == 1) {
                shuffle_cost += c.lhs_input_rec_size;
            }
        }

        // sum the cost for the right side
        for (auto r = 0; r < rhs_tmp.size(); r++) {

            // if it is not assigned we need to fetch it
            if (rhs_record_positions[r * num_nodes + node] == 0 && rhs_tmp[r] == 1) {
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

    // reset lhs tmp
    for(auto l : lhs_tmp_rst) {
        lhs_tmp[l] = false;
    }

    // reset rhs tmp
    for(auto r : rhs_tmp_rst) {
        rhs_tmp[r] = false;
    }

    // return the min node
    return min_node;
}

void pdb::GreedyPlanner::apply_agg(const pdb::GreedyPlanner::agg_plan_t &pl, std::vector<int32_t> &joinGroups,
                                   int32_t aggGroup) {


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
        rhs_record_positions[p.second * num_nodes + pl.node] = true;

        // assign join group
        join_group_positions[jg * num_nodes + pl.node] = true;
    }

    // set the aggregation group position
    aggregation_positions[aggGroup * num_nodes + pl.node] = true;
}

pdb::GreedyPlanner::join_plan_t pdb::GreedyPlanner::try_assign_join_group(std::vector<int32_t> &joinGroups) {

    // total ovehead
    int32_t total_overhead = 0;

    // copy the shuffling costs
    planned_shuffle_cost.resize(side_shuffling_costs.size());
    for(int i = 0; i < side_shuffling_costs.size(); ++i) { planned_shuffle_cost[i] = side_shuffling_costs[i]; }
    auto planned_max_shuffle_cost = max_side_shuffling_cost;

    // copy the join costs
    planned_join_costs.resize(join_projection_costs.size());
    for(int i = 0; i < planned_join_costs.size(); ++i) { planned_join_costs[i] = join_projection_costs[i]; }
    auto planned_max_join_cost = max_join_projection_cost;

    // this keeps track of where we assigned the join groups
    planned_join_assignments.clear();

    // go and assign each join group
    tmpLHSFetches.clear();
    tmpRHSFetches.clear();
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
            auto lhsOffNode = lhs_record_positions[p.first * num_nodes + node] == 0;

            // is the lhs of the join group off this node
            auto rhsOffNode = rhs_record_positions[p.second * num_nodes + node] == 0;

            // if the lhs is of the node mark that we fetched it and update the cost
            if(lhsOffNode) {
                lhs_cost = c.send_coef * c.lhs_input_rec_size;
            }

            // if the rhs if off the node mark that we have fetched and update the cost
            if(rhsOffNode) {
                rhs_cost = c.send_coef * c.rhs_input_rec_size;
            }

            // this is how much the total shuffling cost is
            auto shuffle_cost = lhs_cost + rhs_cost + planned_shuffle_cost[node];
            auto join_cost = planned_join_costs[node] + c.join_cost;

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

        // store the proposed change for lhs
        if(lhs_record_positions[p.first * num_nodes + best_node] == 0) {
            tmpLHSFetches[best_node].emplace_back(p.first);
            lhs_record_positions[p.first * num_nodes + best_node] = true;
        }

        // store the proposed change for lhs
        if(rhs_record_positions[p.second * num_nodes + best_node] == 0) {
            tmpRHSFetches[best_node].emplace_back(p.second);
            rhs_record_positions[p.second * num_nodes + best_node] = true;
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


    // restore the changes we need for the lhs side
    for(int node = 0; node < tmpLHSFetches.size(); node++) {
        for(auto l : tmpLHSFetches[node]) {
            lhs_record_positions[l * num_nodes + node] = 0;
        }
    }

    // restore the changes we need for the rhs side
    for(int node = 0; node < tmpRHSFetches.size(); node++) {
        for(auto r : tmpRHSFetches[node]) {
            rhs_record_positions[r * num_nodes + node] = 0;
        }
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

void pdb::GreedyPlanner::apply_join(const pdb::GreedyPlanner::join_plan_t &pl, std::vector<int32_t> &joinGroups,
                                    int32_t aggGroup) {

    // go through all the join groups
    for(int i = 0; i < joinGroups.size(); ++i) {

        // get the joined lhs and rhs
        auto &p = join_groups[joinGroups[i]];

        // set the assignments
        lhs_record_positions[p.first * num_nodes + pl.join_assignments[i]] = true;
        rhs_record_positions[p.second * num_nodes + pl.join_assignments[i]] = true;

        // assign join group
        join_group_positions[joinGroups[i] * num_nodes + pl.join_assignments[i]] = true;
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

void pdb::GreedyPlanner::print() {

    std::cout << "--aggregation group assignments--\n";
    for(int i = 0; i < num_agg_groups; ++i) {
        std::cout << "group " << i << " assigned to : \n";
        for(int j = 0; j < num_nodes; j++) {
            if(aggregation_positions[i * num_nodes + j ] ) {
                std::cout << j << " node\n";
            }
        }
    }

    std::cout << "\n\n--join group assignments--\n";
    for(int i = 0; i < num_join_groups; ++i) {
        std::cout << "group " << i << " assigned to : \n";
        for(int j = 0; j < num_nodes; j++) {
            if(join_group_positions[i * num_nodes + j ] ) {
                std::cout << j << " node\n";
            }
        }
    }
}
