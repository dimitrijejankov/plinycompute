#pragma once

#include <cstdint>
#include <cstring>
#include <random>
#include <pipeline/Join3KeyPipeline.h>

namespace pdb {

class GeneticJoinPlanner {
public:

  struct genome_t {

    using gene_t = std::vector<int32_t>;

    genome_t() = default;

    explicit genome_t(int32_t size) : genes(size) {}

    int32_t cost{0};

    gene_t genes;
  };

  genome_t generateGenome() {

    // generate the genome
    genome_t join_group_assignments(numJoinRecords);
    for (int i = 0; i < numJoinRecords; ++i) {

      // store the nodes for the aggregation group
      join_group_assignments.genes[i] = node_dist(generator);
    }

    // return the genome
    return std::move(join_group_assignments);
  }

  void init() {

    // generate the initial genomes
    currentGenomes.resize(populationSize);
    for (int i = 0; i < populationSize; ++i) {
      currentGenomes[i] = generateGenome();
    }

    // initialize the surviving genomes (just so we don't have to allocate memory again)
    nextGeneration.resize(populationSize);
    for (int i = 0; i < populationSize; ++i) {
      nextGeneration[i] = genome_t(numJoinRecords);
    }
  }

  void cost(genome_t &genome) {

    // zero out the vector
    for(auto &t : tempRequirements) {
      bzero(t.data(), sizeof(char) * numRecords);
    }

    // go through join records and figure out what we need
    genome.cost = 0;
    for (int i = 0; i < joinedRecords.size(); ++i) {

      // get the join record and the node
      auto &jr = joinedRecords[i];
      auto node = genome.genes[i];

      // get where all the records are
      auto j1 = jr.first;
      auto j2 = jr.second;
      auto j3 = jr.third;
      auto j4 = jr.fourth;
      auto j5 = jr.fifth;
      auto j6 = jr.sixth;
      auto j7 = jr.seventh;
      auto j8 = jr.eigth;

      tempRequirements[node][j1] = true;
      tempRequirements[node][j2] = true;
      tempRequirements[node][j3] = true;
      tempRequirements[node][j4] = true;
      tempRequirements[node][j5] = true;
      tempRequirements[node][j6] = true;
      tempRequirements[node][j7] = true;
      tempRequirements[node][j8] = true;
    }

    for(int record = 0; record < numRecords; ++record) {
      for(int node = 0; node < numNodes; ++node) {
        genome.cost += !this->side_tids[node][record] && tempRequirements[node][record];
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

    // copy elites
    for(int i = 0; i < 5; ++i) {
      for(int j = 0; j < nextGeneration[i].genes.size(); ++j) {
        nextGeneration[i].genes[j] = currentGenomes[i].genes[j];
      }
    }

    // make the next generation
    for(auto it = nextGeneration.begin() + 5; it != nextGeneration.end(); ++it) {

      // get l and r
      auto &l = currentGenomes[selection_dist(generator)];
      auto &r = currentGenomes[selection_dist(generator)];

      // make l and r
      mate(l, r, *it);
    }
  }

  void mate(const genome_t &lhs, const genome_t &rhs, genome_t &result) {

    // set to zero
    bzero(tempStats.data(), sizeof(int32_t) * numNodes);

    // do random crossover
    for (int i = 0; i < numJoinRecords; ++i) {
      bool tmp = crossover_dist(generator);
      result.genes[i] = tmp ? lhs.genes[i] : rhs.genes[i];
      tempStats[result.genes[i]]++;
    }

    // enforce balance
    int32_t numFlipped = 0;
    auto limit = (numJoinRecords / numNodes) + 1;
    for (int i = 0; i < numJoinRecords; ++i) {

      auto node = result.genes[i];
      if (tempStats[node] > limit) {
        for (int j = 0; j < numNodes; ++j) {
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

  void mutate() {

    for (int i = 5; i < populationSize; ++i) {

      // do we mutate or not?
      float t = mutation_dist(generator);
      if (t < mutation_rate) {

        // get the gene
        auto i_g = join_group_dist(generator);
        auto j_g = join_group_dist(generator);

        // swap the genes
        auto tmp = nextGeneration[i].genes[i_g];
        nextGeneration[i].genes[i_g] = nextGeneration[i].genes[j_g];
        nextGeneration[i].genes[j_g] = tmp;
      }
    }
  }

  auto &getResult() {

    // sort the genomes
    std::sort(currentGenomes.begin(), currentGenomes.end(), [](const genome_t &lhs, const genome_t &rhs) {
      return lhs.cost < rhs.cost;
    });

    return currentGenomes.front().genes;
  }

  void finish() {
    std::swap(currentGenomes, nextGeneration);
  }

  GeneticJoinPlanner(int32_t numRecords,
                     int32_t numJoinRecords,
                     int32_t numNodes,
                     int32_t sideRecordSize,
                     std::vector<std::vector<bool>> &side_tids,
                     std::vector<Join3KeyPipeline::joined_record> &joinedRecords,
                     int32_t populationSize);

  void run(int numIter) {

    // init the planner
    init();

    // run for a number of iterations
    for(int i = 0; i < numIter; ++i) {

      selection();
      print();
      crossover();
      mutate();
      finish();
    }
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

  std::vector<int32_t> tempStats;
  std::vector<std::vector<char>> tempRequirements;
  std::vector<genome_t> currentGenomes;
  std::vector<genome_t> nextGeneration;

  int32_t numRecords;
  int32_t numJoinRecords;
  int32_t numNodes;
  int32_t sideRecordSize;

  const std::vector<std::vector<bool>> &side_tids;
  const std::vector<Join3KeyPipeline::joined_record> &joinedRecords;

  int32_t populationSize;
  const float mutation_rate = 0.05;

  std::default_random_engine generator;

  std::uniform_int_distribution<int32_t> node_dist;
  std::uniform_int_distribution<int32_t> selection_dist;
  std::uniform_int_distribution<int32_t> join_group_dist;
  std::uniform_int_distribution<int32_t> crossover_dist{0, 1};
  std::uniform_real_distribution<float> mutation_dist{0.0, 1.0};
};

}