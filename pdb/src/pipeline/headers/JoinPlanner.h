#pragma once

#include <unordered_map>
#include <vector>
#include <Join3KeyPipeline.h>

namespace pdb {

class JoinPlanner {
public:

  JoinPlanner(uint32_t numNodes,
              uint32_t numThreads,
              std::unordered_map<Join3KeyPipeline::key, pair<int32_t, int32_t>, Join3KeyPipeline::HashFunction> &nodeRecords0,
              std::unordered_map<Join3KeyPipeline::key, pair<int32_t, int32_t>, Join3KeyPipeline::HashFunction> &nodeRecords1,
              std::unordered_map<Join3KeyPipeline::key, pair<int32_t, int32_t>, Join3KeyPipeline::HashFunction> &nodeRecords2,
              std::vector<Join3KeyPipeline::joined_record> &joined,
              std::vector<std::vector<int32_t>> &aggGroups);

  void doPlanning(const PDBPageHandle &page);

  // these are the node records
  std::unordered_map<Join3KeyPipeline::key, pair<int32_t, int32_t>, Join3KeyPipeline::HashFunction> &nodeRecords0;
  std::unordered_map<Join3KeyPipeline::key, pair<int32_t, int32_t>, Join3KeyPipeline::HashFunction> &nodeRecords1;
  std::unordered_map<Join3KeyPipeline::key, pair<int32_t, int32_t>, Join3KeyPipeline::HashFunction> &nodeRecords2;

  // the joined records
  std::vector<Join3KeyPipeline::joined_record> &joined;

  // the aggregation groups
  std::vector<std::vector<int32_t>> &aggGroups;

  // the number of nodes for planning
  uint32_t numNodes;

  // the number of threads
  uint32_t numThreads;
};

}
