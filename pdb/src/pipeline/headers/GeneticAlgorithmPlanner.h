#pragma once

#include <random>

namespace pdb {

struct GeneticAlgorithmPlanner {

  struct genome_t {

    using gene_t = std::vector<int32_t>;

    genome_t() = default;

    explicit genome_t(int32_t size) : genes(size) {}

    int32_t cost{0};

    gene_t genes;
  };

  GeneticAlgorithmPlanner(int32_t num_lhs_records,
                          int32_t num_rhs_records,
                          int32_t num_nodes,
                          int32_t num_agg_groups,
                          int32_t lhs_record_size,
                          int32_t rhs_record_size,
                          std::vector<std::vector<bool>> lhs_record_positions,
                          std::vector<std::vector<bool>> rhs_record_positions,
                          std::vector<std::vector<std::pair<int32_t, int32_t>>> aggregation_groups,
                          int32_t init_population_size) : node_dist(0, num_nodes - 1),
                                                          selection_dist(0, (init_population_size - 1) / 2),
                                                          aggregation_group_dist(0, num_agg_groups - 1),
                                                          num_lhs_records(num_lhs_records),
                                                          num_rhs_records(num_rhs_records),
                                                          num_nodes(num_nodes),
                                                          num_agg_groups(num_agg_groups),
                                                          lhs_record_size(lhs_record_size),
                                                          rhs_record_size(rhs_record_size),
                                                          init_population_size(init_population_size),
                                                          lhs_record_positions(std::move(lhs_record_positions)),
                                                          rhs_record_positions(std::move(rhs_record_positions)),
                                                          aggregation_groups(std::move(aggregation_groups)) {

    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    tempStats.resize(num_nodes);
  }

  genome_t generateGenome() {

    // generate the genome
    genome_t aggregation_group_assignments(num_agg_groups);
    for (int i = 0; i < num_agg_groups; ++i) {

      // store the nodes for the aggregation group
      aggregation_group_assignments.genes[i] = node_dist(generator);
    }

    return std::move(aggregation_group_assignments);
  }

  void init() {

    // generate the initial genomes
    currentGenomes.resize(init_population_size);
    for (int i = 0; i < init_population_size; ++i) {
      currentGenomes[i] = generateGenome();
    }

    // initialize the surviving genomes (just so we don't have to allocate memory again)
    nextGeneration.resize(init_population_size);
    for (int i = 0; i < init_population_size; ++i) {
      nextGeneration[i] = genome_t(num_agg_groups);
    }
  }

  void cost(genome_t &genome) {

    genome.cost = 0;
    for (int i = 0; i < num_agg_groups; i++) {

      // get the node it belongs to
      auto &node = genome.genes[i];

      // go through each aggregation group
      for (const auto &g : aggregation_groups[i]) {
        genome.cost += 1 - lhs_record_positions[g.first][node];
      }
      for (const auto &g : aggregation_groups[i]) {
        genome.cost += 1 - rhs_record_positions[g.second][node];
      }
    }
  }

  void selection() {

    // we calculate the s
    for (auto &g : currentGenomes) {

      // cost the genome
      cost(g);
    }

    //
    std::sort(currentGenomes.begin(), currentGenomes.end(), [](const genome_t &lhs, const genome_t &rhs) {
      return lhs.cost < rhs.cost;
    });
  }

  void crossover() {

    // make the next generation
    for (auto &i : nextGeneration) {

      // get l and r
      auto &l = currentGenomes[selection_dist(generator)];
      auto &r = currentGenomes[selection_dist(generator)];

      // make l and r
      mate(l, r, i);
    }
  }

  void mutation() {
    // mutate all
    for (auto &i : nextGeneration) {
      mutate(i);
    }
  }

  void finish() {
    std::swap(currentGenomes, nextGeneration);
  }

  void run(int numIter) {

    // init the planner
    init();

    // run for a number of iterations
    for(int i = 0; i < numIter; ++i) {

      selection();
      crossover();
      mutation();
      finish();
      print();
    }
  }

  auto &getResult() {

    // sort the genomes
    std::sort(currentGenomes.begin(), currentGenomes.end(), [](const genome_t &lhs, const genome_t &rhs) {
      return lhs.cost < rhs.cost;
    });

    return currentGenomes.front().genes;
  }

  void print() {
    for (auto &g : currentGenomes) {
      cost(g);
      std::cout << g.cost << " : |";

      for (int gene : g.genes) {
        std::cout << gene << "| ";
      }

      std::cout << '\n';
    }
    std::cout << "next gen\n";
  }

  void mate(const genome_t &lhs, const genome_t &rhs, genome_t &result) {

    // set to zero
    bzero(tempStats.data(), sizeof(int32_t) * num_nodes);

    // do random crossover
    for (int i = 0; i < num_agg_groups; ++i) {
      bool tmp = crossover_dist(generator);
      result.genes[i] = tmp ? lhs.genes[i] : rhs.genes[i];
      tempStats[result.genes[i]]++;
    }

    // enforce balance
    int32_t numFlipped = 0;
    auto limit = (num_agg_groups / num_nodes) + 1;
    for (int i = 0; i < num_agg_groups; ++i) {

      auto node = result.genes[i];
      if (tempStats[node] > limit) {
        for (int j = 0; j < num_nodes; ++j) {
          if (tempStats[j] < limit) {
            result.genes[i] = j;
            tempStats[j]++;
            tempStats[node]--;
            numFlipped++;
            break;
          }
        }
      }
    }
  }

  void mutate(genome_t &lhs) {

    for (int i = 0; i < init_population_size; ++i) {

      // do we mutate or not?
      float t = mutation_dist(generator);
      if (t < mutation_rate) {

        // get the gene
        auto i_g = aggregation_group_dist(generator);
        auto j_g = aggregation_group_dist(generator);

        // swap the genes
        auto tmp = nextGeneration[i].genes[i_g];
        nextGeneration[i].genes[i_g] = nextGeneration[i].genes[j_g];
        nextGeneration[i].genes[j_g] = tmp;
      }
    }
  }

  std::vector<int32_t> tempStats;
  std::vector<std::vector<bool>> lhs_record_positions;
  std::vector<std::vector<bool>> rhs_record_positions;
  std::vector<std::vector<std::pair<int32_t, int32_t>>> aggregation_groups;
  std::vector<genome_t> currentGenomes;
  std::vector<genome_t> nextGeneration;
  std::default_random_engine generator;

  std::uniform_int_distribution<int32_t> node_dist;
  std::uniform_int_distribution<int32_t> selection_dist;
  std::uniform_int_distribution<int32_t> aggregation_group_dist;
  std::uniform_int_distribution<int32_t> crossover_dist{0, 1};
  std::uniform_real_distribution<float> mutation_dist{0.0, 1.0};

  const float mutation_rate = 0.05;

  int32_t num_lhs_records;
  int32_t num_rhs_records;
  int32_t num_nodes;
  int32_t num_agg_groups;
  int32_t lhs_record_size;
  int32_t rhs_record_size;

  int32_t init_population_size;
};

}