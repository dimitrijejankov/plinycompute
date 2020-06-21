#pragma once

#include "PDBAnonymousPageSet.h"
#include "PDBVector.h"
#include "PipJoinAggPlanResult.h"

namespace pdb {

class JoinAggPlanner {
 public:

  explicit JoinAggPlanner(const pdb::PDBAnonymousPageSetPtr &joinAggPageSet,
                          uint32_t numNodes,
                          uint32_t numThreads,
                          const PDBPageHandle &pageToStore);

  void doPlanning();

  void print(const Handle<PipJoinAggPlanResult> &planResult);

  bool isLocalAggregation();

 private:

  // we use this to deduplicate the join groups
  struct join_group_hasher {
    size_t operator()(const pair<int32_t, int32_t> &p) const {
      auto hash1 = hash<int32_t>{}(p.first);
      auto hash2 = hash<int32_t>{}(p.second);
      return hash1 ^ hash2;
    }
  };

  void doAggFirstPlanning(const std::vector<char> &lhsRecordPositions,
                          const std::vector<char> &rhsRecordPositions,
                          const std::vector<std::vector<int32_t>> &aggregationGroups,
                          const std::vector<PipJoinAggPlanResult::JoinedRecord> &joinGroups);

  void doJoinFirstPlanning(const std::vector<char> &lhsRecordPositions, const std::vector<char> &rhsRecordPositions,
                           const std::vector<std::vector<int32_t>> &aggregationGroups,
                           const std::vector<PipJoinAggPlanResult::JoinedRecord> &joinGroups);

  void doFullPlanning(const std::vector<char> &lhsRecordPositions, const std::vector<char> &rhsRecordPositions,
                      const std::vector<std::vector<int32_t>> &aggregationGroups,
                      const std::vector<PipJoinAggPlanResult::JoinedRecord> &joinGroups);

  long agg_first_time{};
  long join_first_time{};
  long full_first_time{};

  // the page containing the page
  pdb::PDBPageHandle joinAggPageSet;

  // the first integer is the join key value identifier, the second value is the node it comes from
  using TIDType = std::pair<uint32_t, int32_t>;

  // this is a vector of keys that are joined to form an aggregation group
  using TIDVector = Vector<std::pair<TIDType, TIDType>>;

  // each aggregation group is identified by an unsigned integer
  // this maps maps the aggregation group to all the keys that are joined to form it
  using TIDIndexMap = Map<uint32_t, TIDVector>;

  // the input page
  PDBPageHandle inputPage;

  // the the map
  Handle<TIDIndexMap> aggGroups;

  // the number of nodes
  int32_t numNodes;

  //  the number of threads for each node
  int32_t numThreads;

  // the page where we store the result
  PDBPageHandle pageToStore;

  // the algorithm identifier
  enum class AlgorithmID : int32_t {
    AGG_FIRST_ONLY = 0,
    JOIN_FIRST_ONLY = 1,
    FULL = 2
  };

  // selects the algorithm we are gonna use
  AlgorithmID selectAlgorithm();

  // the algorithm we selected
  AlgorithmID selectedAlgorithm;

  // how many planners have finished we will run three at the same time
  atomic_int32_t num_finished;

  // the planning costs each planner could come up with
  int32_t planning_costs[3];
};

}