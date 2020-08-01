#pragma once

#include <random>

namespace pdb {

class GeneticAggGroupPlanner {
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

  struct planning_result {

    planning_result(std::vector<char> agg_group_assignments,
                    std::vector<char> join_groups_node)
        : agg_group_assignments(std::move(agg_group_assignments)),
          join_groups_to_node(std::move(join_groups_node)) {}

    // tells us what node the aggregation group was assigned
    std::vector<char> agg_group_assignments;

    // tells us
    std::vector<char> join_groups_to_node;
  };

  GeneticAggGroupPlanner(int32_t initPopulationSize,
                   int32_t numNodes,
                   int32_t numAggGroups,
                   int32_t numJoinGroups,
                   int32_t numLhsRecords,
                   int32_t numRhsRecords,
                   costs_t costs,
                   std::vector<char> lhsRecordPositions,
                   std::vector<char> rhsRecordPositions,
                   std::vector<std::vector<int32_t>> aggregationGroups,
                   std::vector<PipJoinAggPlanResult::JoinedRecord> joinGroups)
      : init_population_size(initPopulationSize),
        num_nodes(numNodes),
        num_agg_groups(numAggGroups),
        num_join_groups(numJoinGroups),
        num_lhs_records(numLhsRecords),
        num_rhs_records(numRhsRecords),
        c(costs),
        lhs_record_positions(std::move(lhsRecordPositions)),
        rhs_record_positions(std::move(rhsRecordPositions)),
        aggregation_groups(std::move(aggregationGroups)),
        join_groups(std::move(joinGroups)) {

  }

  struct genome_t {

    genome_t() = default;

    genome_t(int32_t numNodes, int32_t numLHS, int32_t numRHS, int32_t numJoin, int32_t numAgg) {

      // init the lhs fetches for each node
      lhs_fetch.resize(numNodes);
      for (auto &l : lhs_fetch) {
        l.resize(numLHS);
      }

      // init the rhs fetches for each node
      rhs_fetch.resize(numNodes);
      for (auto &r : rhs_fetch) {
        r.resize(numRHS);
      }

      // init the group assignments
      join_group_assignments.resize(numJoin);

      // init the join group fetches
      join_group_fetch.resize(numNodes);
      for (auto &j : join_group_fetch) {
        j.resize(numJoin);
      }

      // init the aggregation group assignments
      agg_group_assignments.resize(numAgg);

      // init the temp memory
      tmp.resize(numNodes);
    }

    // where do we get the lhs [node, lhs_record_tid], the value indicates what node we are fetching it from
    std::vector<std::vector<char>> lhs_fetch;

    // where do we get the rhs [node, rhs_record_tid], the value indicates what node we are fetching it from
    std::vector<std::vector<char>> rhs_fetch;

    // where the join groups are assigned [join_group_tid, node]
    std::vector<char> join_group_assignments;

    // where do we fetch the groups from [node, join_group_tid], the value indicates what node we are fetching it from
    std::vector<std::vector<char>> join_group_fetch;

    // this tells us where the aggregation groups were assigned [aggregation_group] to node
    std::vector<char> agg_group_assignments;

    // we use this while costing it is the size of node
    std::vector<int32_t> tmp;

    // the cost of the genome
    int64_t cost{};
  };

  genome_t generateGenome() {

    // make a seed
    int32_t seed = std::chrono::system_clock::now().time_since_epoch().count();

    // generate the genome
    genome_t genome(num_nodes, num_lhs_records, num_rhs_records, num_join_groups, num_agg_groups);

    // store the nodes for the aggregation group
    for (auto &agg : genome.agg_group_assignments) {
      agg = fast_rand_int(seed) % num_nodes;
    }

    // make the join assigments
    for (auto &jg : genome.join_group_assignments) {
      jg = fast_rand_int(seed) % num_nodes;
    }

    // generate the assignments
    generate_assignments(genome);

    // return the genome
    return std::move(genome);
  }


  planning_result get_result();

  void init() {

    // generate the initial genomes
    currentGeneration.resize(init_population_size);
    for (int i = 0; i < init_population_size; ++i) {
      currentGeneration[i] = generateGenome();
    }

    // initialize the surviving genomes (just so we don't have to allocate memory again)
    nextGeneration.resize(init_population_size);
    for (int i = 0; i < init_population_size; ++i) {
      nextGeneration[i] = generateGenome();
    }
  }

  void cost(genome_t &g) {

    /// We need to calculate this :
    /// max(shuffle_send, shuffle_recv) + max(join_send, join_recv) +
    /// max(join_projection) + max(agg_projection)

    /// 1. for each node calculate the shuffle cost (shuffle_send, shuffle_recv)

    // do it for the lhs
    int32_t lhs_recv_cost = 0;
    int32_t lhs_send_cost = 0;
    for (int32_t node = 0; node < g.lhs_fetch.size(); ++node) {

      // we keep the values for this node
      int32_t tmp_send = 0;
      int32_t tmp_recv = 0;

      // get the join groups for this node
      auto &lhs_fetch = g.lhs_fetch[node];

      // go through each record we need to fetch
      for (char r : lhs_fetch) {
        tmp_recv += node != r;
        tmp_send += node == r;
        tmp_recv -= r != -1;
        tmp_send -= r != -1;
      }

      lhs_recv_cost = std::max(tmp_send, lhs_recv_cost);
      lhs_send_cost = std::max(tmp_recv, lhs_send_cost);
    }

    // do it for the rhs
    int32_t rhs_recv_cost = 0;
    int32_t rhs_send_cost = 0;
    for (int32_t node = 0; node < g.rhs_fetch.size(); ++node) {

      // we keep the values for this node
      int32_t tmp_send = 0;
      int32_t tmp_recv = 0;

      // get the join groups for this node
      auto &rhs_fetch = g.rhs_fetch[node];

      // go through each record we need to fetch
      for (char r : rhs_fetch) {
        tmp_recv += node != r;
        tmp_send += node == r;
        tmp_recv -= r == -1;
        tmp_send -= r == -1;
      }

      rhs_recv_cost = std::max(tmp_send, rhs_recv_cost);
      rhs_send_cost = std::max(tmp_recv, rhs_send_cost);
    }

    /// 2. for each node calculate, the join groups we need to send

    // do it for the rhs
    int32_t join_groups_recv_cost = 0;
    int32_t join_groups_send_cost = 0;
    for (int32_t node = 0; node < g.join_group_fetch.size(); ++node) {

      // we keep the values for this node
      int32_t tmp_send = 0;
      int32_t tmp_recv = 0;

      // get the join groups for this node
      auto &join_groups_fetch = g.join_group_fetch[node];

      // go through each record we need to fetch
      for (char r : join_groups_fetch) {
        tmp_recv += node != r;
        tmp_send += node == r;
        tmp_recv -= r == -1;
        tmp_send -= r == -1;
      }

      // figure out the new max
      join_groups_recv_cost = std::max(tmp_send, join_groups_recv_cost);
      join_groups_send_cost = std::max(tmp_recv, join_groups_send_cost);
    }

    /// 3. go through all the nodes to find the largest time to do the join projection

    std::fill(g.tmp.begin(), g.tmp.end(), 0);
    for (auto &node : g.join_group_assignments) {

      // increment since it is assigned to this node
      g.tmp[node]++;
    }
    int32_t join_projection = *std::max_element(g.tmp.begin(), g.tmp.end());

    /// 4. go through all the nodes and find the largest aggregation cost

    std::fill(g.tmp.begin(), g.tmp.end(), 0);
    for (auto &node : g.agg_group_assignments) {

      // sum them up
      g.tmp[node]++;
    }
    int32_t aggregation_projection = *std::max_element(g.tmp.begin(), g.tmp.end());

    // calculate the cost
    g.cost =
        c.send_coef *
            std::max<int64_t>(c.lhs_input_rec_size * lhs_send_cost + c.rhs_input_rec_size * rhs_send_cost,
                              c.lhs_input_rec_size * lhs_recv_cost + c.rhs_input_rec_size * rhs_recv_cost) +
            c.send_coef * c.join_rec_size * std::max<int64_t>(join_groups_recv_cost, join_groups_send_cost) +
            c.join_cost * join_projection + c.agg_cost * aggregation_projection;
  }

  static inline bool fast_rand_bool(int32_t &g_seed) {
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x1;
  }

  static inline int fast_rand_int(int32_t &g_seed) {
    g_seed = (214013 * g_seed + 2531011);
    return g_seed >> 16 & 0x1FFFFFF;
  }

  void mate(genome_t &out, genome_t &p1, genome_t &p2, int32_t seed) {

    /// 1. do crossover for join
    for (auto i = 0; i < out.join_group_assignments.size(); ++i) {

      // flip a coin
      if (fast_rand_bool(seed) % 2 == 0) {

        std::swap(out.join_group_assignments[i], p1.join_group_assignments[i]);
      } else {
        std::swap(out.join_group_assignments[i], p2.join_group_assignments[i]);
      }
    }

    /// 2. do crossover for aggregation
    for (auto i = 0; i < out.agg_group_assignments.size(); ++i) {

      // flip a coin
      if (fast_rand_bool(seed) % 2 == 0) {

        std::swap(out.agg_group_assignments[i], p1.agg_group_assignments[i]);
      } else {
        std::swap(out.agg_group_assignments[i], p2.agg_group_assignments[i]);
      }
    }

    /// 3. do mutation on the join
    for (auto &join_group_assignment : out.join_group_assignments) {

      // chance for mutation is 5% mutation
      if (fast_rand_int(seed) % 1000 < 10) {
        join_group_assignment = fast_rand_int(seed) % num_nodes;
      }
    }

    /// 4. do mutation on aggregation
    for (char &agg_group_assignment : out.agg_group_assignments) {

      // chance for mutation is 5%
      if (fast_rand_int(seed) % 1000 < 10) {
        agg_group_assignment = (char) (fast_rand_int(seed) % num_nodes);
      }
    }

    generate_assignments(out);
  }

  void selection() {

    // we calculate the s
    //#pragma omp parallel for
    for (auto &g : currentGeneration) {

      // cost the genome
      cost(g);
    }

    //
    std::sort(currentGeneration.begin(), currentGeneration.end(), [](const genome_t &lhs, const genome_t &rhs) {
      return lhs.cost < rhs.cost;
    });
  }

  void crossover_mutation() {

    // make a seed
    int32_t seed = std::chrono::system_clock::now().time_since_epoch().count();

    // make the next generation
    for (int i = 0; i < elite_num; ++i) {
      nextGeneration[i] =  currentGeneration[i];
    }

    //#pragma omp parallel for
    for (int i = elite_num; i < nextGeneration.size(); ++i) {

      // get l and r
      auto &l = currentGeneration[fast_rand_int(seed) % init_population_size];
      auto &r = currentGeneration[fast_rand_int(seed) % init_population_size];

      // make l and r
      mate(nextGeneration[i], l, r, seed);
    }

  }

  void finish() {
    std::swap(currentGeneration, nextGeneration);
  }

  void run(int numIter) {

    // init the planner
    init();

    // run for a number of iterations
    for (int i = 0; i < numIter; ++i) {

      selection();
      crossover_mutation();
      finish();

      //print();
    }

    auto &best = currentGeneration[0];
    for (int i = 0; i < num_join_groups; ++i) {
      std::cout << "Join group " << i << " assigned to " << (int) best.join_group_assignments[i] << " Node\n";
    }

    for (int i = 0; i < num_agg_groups; ++i) {
      auto agg = best.agg_group_assignments[i];

      std::cout << "Group " << i << " assigned to " << (int) agg << " node\n";

      // join groups are on
      for (auto jg : aggregation_groups[i]) {
        std::cout << "Join group " << jg << " on " << (int) best.join_group_assignments[jg] << '\n';
      }
    }

    std::cout << "Cost : " << currentGeneration[0].cost << '\n';
  }

  void generate_assignments(genome_t &out) const {

    // reset all fetches to -1
    for (int32_t i = 0; i < num_nodes; ++i) {
      std::fill(out.lhs_fetch[i].begin(), out.lhs_fetch[i].end(), -1);
      std::fill(out.rhs_fetch[i].begin(), out.rhs_fetch[i].end(), -1);
      std::fill(out.join_group_fetch[i].begin(), out.join_group_fetch[i].end(), -1);
    }

    /// 5. figure out where to fetch stuff

    //
    for (auto i = 0; i < join_groups.size(); ++i) {

      // where the join node was assigned
      auto join_node = out.join_group_assignments[i];

      // left and right record
      auto lhs = join_groups[i].lhsTID;
      auto rhs = join_groups[i].rhsTID;

      // the nodes where they are assigned
      auto lhs_node = lhs_record_positions[lhs];
      auto rhs_node = rhs_record_positions[rhs];

      // set the nodes we are fetching from
      out.lhs_fetch[join_node][lhs] = lhs_node;
      out.rhs_fetch[join_node][rhs] = rhs_node;
    }

    // go through the aggregation groups
    for (auto i = 0; i < aggregation_groups.size(); ++i) {

      // the aggregation group node
      auto aggregation_node = out.agg_group_assignments[i];

      // go through the aggregation groups and figure out where we need to fetch them
      for (auto jg : aggregation_groups[i]) {
        out.join_group_fetch[aggregation_node][jg] = out.join_group_assignments[jg];
      }
    }
  }

  const int elite_num = 5;

  int32_t init_population_size;
  int32_t num_nodes = 3;
  int32_t num_agg_groups = 16;
  int32_t num_join_groups = 16;
  int32_t num_lhs_records;
  int32_t num_rhs_records;

  std::vector<genome_t> currentGeneration;
  std::vector<genome_t> nextGeneration;

  // [node][join_group]
  std::vector<char> lhs_record_positions;

  // [node][join_group]
  std::vector<char> rhs_record_positions;

  std::vector<std::vector<int32_t>> aggregation_groups;

  std::vector<PipJoinAggPlanResult::JoinedRecord> join_groups;

  costs_t c;
};

GeneticAggGroupPlanner::planning_result GeneticAggGroupPlanner::get_result() {
  return GeneticAggGroupPlanner::planning_result(std::move(currentGeneration[0].agg_group_assignments),
                                                 std::move(currentGeneration[0].join_group_assignments));
}

}