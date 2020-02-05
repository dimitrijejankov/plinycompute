#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <GeneticAlgorithmPlanner.h>


TEST(TestPlanner, Test1) {

  int32_t num_lhs_records = 16;
  int32_t num_rhs_records = 16;
  int32_t num_nodes = 3;
  int32_t num_agg_groups = 16;
  int32_t lhs_record_size = 1;
  int32_t rhs_record_size = 1;

  // node                                                  0      1      2
  std::vector<std::vector<bool>> lhs_record_positions = {{true, false, false},  // record 0
                                                         {false, true, false},  // record 1
                                                         {false, false, true},  // record 2
                                                         {false, true, false},  // record 3
                                                         {true, false, false},  // record 4
                                                         {false, true, false},  // record 5
                                                         {false, false, true},  // record 6
                                                         {false, true, false},  // record 7
                                                         {true, false, false},  // record 8
                                                         {false, true, false},  // record 9
                                                         {false, false, true},  // record 10
                                                         {false, true, false},  // record 11
                                                         {true, false, false},  // record 12
                                                         {false, true, false},  // record 13
                                                         {false, false, true},  // record 14
                                                         {false, true, false}}; // record 15

  // node                                                  0      1      2
  std::vector<std::vector<bool>> rhs_record_positions = {{true, false, false},  // record 0
                                                         {true, false, false},  // record 1
                                                         {true, false, false},  // record 2
                                                         {false, true, false},  // record 3
                                                         {false, true, false},  // record 4
                                                         {false, true, false},  // record 5
                                                         {false, false, true},  // record 6
                                                         {false, false, true},  // record 7
                                                         {false, false, true},  // record 8
                                                         {false, true, false},  // record 9
                                                         {false, true, false},  // record 10
                                                         {false, true, false},  // record 11
                                                         {true, false, false},  // record 12
                                                         {true, false, false},  // record 13
                                                         {true, false, false},  // record 14
                                                         {false, true, false}}; // record 15

  // aggregation groups
  std::vector<std::vector<std::pair<int32_t, int32_t>>> aggregation_groups = {{{0, 0}, {1, 4}, {2, 8}, {3, 12}},
                                                                              {{0, 1}, {1, 5}, {2, 9}, {3, 13}},
                                                                              {{0, 2}, {1, 6}, {2, 10}, {3, 14}},
                                                                              {{0, 3}, {1, 7}, {2, 11}, {3, 15}},
                                                                              {{4, 0}, {5, 4}, {6, 8}, {7, 12}},
                                                                              {{4, 1}, {5, 5}, {6, 9}, {7, 13}},
                                                                              {{4, 2}, {5, 6}, {6, 10}, {7, 14}},
                                                                              {{4, 3}, {5, 7}, {6, 11}, {7, 15}},
                                                                              {{8, 0}, {9, 4}, {10, 9}, {11, 12}},
                                                                              {{8, 1}, {9, 5}, {10, 9}, {11, 13}},
                                                                              {{8, 2}, {9, 6}, {10, 10}, {11, 14}},
                                                                              {{8, 3}, {9, 7}, {10, 11}, {11, 15}},
                                                                              {{12, 0}, {13, 4}, {14, 8}, {15, 12}},
                                                                              {{12, 1}, {13, 5}, {14, 9}, {15, 13}},
                                                                              {{12, 2}, {13, 6}, {14, 10}, {15, 14}},
                                                                              {{12, 3}, {13, 7}, {14, 11}, {15, 15}}};

  pdb::GeneticAlgorithmPlanner planner(16, 16, 3, 16, 1, 1, lhs_record_positions, rhs_record_positions, aggregation_groups, 40);

  planner.init();

  for(int i = 0; i < 1000; ++i) {

    planner.selection();
    planner.crossover();
    planner.mutation();
    planner.finish();
    planner.print();
  }


}
