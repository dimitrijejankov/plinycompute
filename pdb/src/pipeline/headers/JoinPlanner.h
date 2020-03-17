#pragma once

#include <unordered_map>
#include <vector>
#include <EightWayJoinPipeline.h>

namespace pdb {

class JoinPlanner {
public:

  JoinPlanner(uint32_t numNodes,
              uint32_t numThreads,
              std::unordered_map<EightWayJoinPipeline::key, pair<int32_t, int32_t>, EightWayJoinPipeline::HashFunction> &nodeRecords,
              std::vector<EightWayJoinPipeline::joined_record> &joined);

  void doPlanning(const PDBPageHandle &page);

  // these are the node records
  std::unordered_map<EightWayJoinPipeline::key, pair<int32_t, int32_t>, EightWayJoinPipeline::HashFunction> &nodeRecords;

  // the joined records
  std::vector<EightWayJoinPipeline::joined_record> &joined;

  // the number of nodes for planning
  uint32_t numNodes;

  // the number of threads
  uint32_t numThreads;
};

}
