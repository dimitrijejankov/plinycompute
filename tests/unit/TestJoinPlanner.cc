#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <EightWayJoinPipeline.h>
#include <GeneticJoinPlanner.h>
#include <JoinPlanner.h>

namespace pdb {

TEST(TestPlanner, Test1) {

  const uint32_t numX = 4;
  const uint32_t numY = 4;
  const uint32_t numZ = 4;
  const uint32_t numNodes = 4;
  const uint32_t numThreads = 4;

  std::vector<std::vector<int32_t>> tmp = {{22, 26, 6, 10, 21, 25, 5, 9},
                                           {59, 63, 43, 47, 58, 62, 42, 46},
                                           {21, 25, 5, 9, 20, 24, 4, 8},
                                           {55, 59, 39, 43, 54, 58, 38, 42},
                                           {54, 58, 38, 42, 53, 57, 37, 41},
                                           {53, 57, 37, 41, 52, 56, 36, 40},
                                           {58, 62, 42, 46, 57, 61, 41, 45},
                                           {50, 54, 34, 38, 49, 53, 33, 37},
                                           {57, 61, 41, 45, 56, 60, 40, 44},
                                           {49, 53, 33, 37, 48, 52, 32, 36},
                                           {51, 55, 35, 39, 50, 54, 34, 38},
                                           {23, 27, 7, 11, 22, 26, 6, 10},
                                           {25, 29, 9, 13, 24, 28, 8, 12},
                                           {26, 30, 10, 14, 25, 29, 9, 13},
                                           {33, 37, 17, 21, 32, 36, 16, 20},
                                           {27, 31, 11, 15, 26, 30, 10, 14},
                                           {17, 21, 1, 5, 16, 20, 0, 4},
                                           {18, 22, 2, 6, 17, 21, 1, 5},
                                           {19, 23, 3, 7, 18, 22, 2, 6},
                                           {37, 41, 21, 25, 36, 40, 20, 24},
                                           {38, 42, 22, 26, 37, 41, 21, 25},
                                           {39, 43, 23, 27, 38, 42, 22, 26},
                                           {41, 45, 25, 29, 40, 44, 24, 28},
                                           {42, 46, 26, 30, 41, 45, 25, 29},
                                           {43, 47, 27, 31, 42, 46, 26, 30},
                                           {34, 38, 18, 22, 33, 37, 17, 21},
                                           {35, 39, 19, 23, 34, 38, 18, 22}};

  std::vector<EightWayJoinPipeline::joined_record> joined;
  joined.reserve(tmp.size());
  for(auto &t : tmp) {
    joined.emplace_back(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
  }


  std::unordered_map<EightWayJoinPipeline::key, pair<int32_t, int32_t>, EightWayJoinPipeline::HashFunction> nodeRecords;

  EightWayJoinPipeline::key key;

  // fill the vector up
  int tid = 0;
  for (int32_t x = 0; x < numX; x++) {
    for (int32_t y = 0; y < numY; y++) {
      for (int32_t z = 0; z < numZ; z++) {
        std::get<0>(key) = x;
        std::get<1>(key) = y;
        std::get<2>(key) = z;
        nodeRecords[key] = std::make_pair(tid, tid % numNodes);
        tid++;
      }
    }
  }

  // make the join planner
  JoinPlanner planner(numNodes,
                      numThreads,
                      nodeRecords,
                      joined);

  planner.doPlanning();

}

}

