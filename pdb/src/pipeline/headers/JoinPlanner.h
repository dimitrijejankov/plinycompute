#pragma once

#include <unordered_map>
#include <vector>
#include <Join3KeyPipeline.h>

namespace pdb {

class JoinPlanner {
public:

  JoinPlanner(uint32_t numNodes,
              uint32_t numThreads,
              std::unordered_map<Join3KeyPipeline::key, pair<int32_t, int32_t>, Join3KeyPipeline::HashFunction> &nodeRecords,
              std::vector<Join3KeyPipeline::joined_record> &joined);

  void doPlanning(const PDBPageHandle &page);

  // these are the node records
  std::unordered_map<Join3KeyPipeline::key, pair<int32_t, int32_t>, Join3KeyPipeline::HashFunction> &nodeRecords;

  // the joined records
  std::vector<Join3KeyPipeline::joined_record> &joined;

  // the number of nodes for planning
  uint32_t numNodes;

  // the number of threads
  uint32_t numThreads;
};

}
