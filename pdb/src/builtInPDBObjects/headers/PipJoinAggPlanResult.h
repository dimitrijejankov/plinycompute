#pragma once

#include <Handle.h>
#include <PDBVector.h>

// PRELOAD %PipJoinAggPlanResult%

namespace pdb {

// encapsulates a request to add data to a set in storage
class PipJoinAggPlanResult : public Object {

public:

  PipJoinAggPlanResult() = default;

  explicit PipJoinAggPlanResult(int32_t numNodes) {

    // init stuff
    this->leftToNode = pdb::makeObject<PipJoinAggPlanResult::JoinTIDToNode>();
    this->rightToNode = pdb::makeObject<PipJoinAggPlanResult::JoinTIDToNode>();
    this->aggToNode = pdb::makeObject<PipJoinAggPlanResult::AggGroupToNode >();
    this->joinGroupsPerNode = pdb::makeObject<PipJoinAggPlanResult::JoinGroupsPerNode>(numNodes, numNodes);
  };


  ~PipJoinAggPlanResult() = default;

  ENABLE_DEEP_COPY

  // join to node map
  using JoinTIDToNode = Map<uint32_t, Vector<bool>>;

  // aggregation group to node map1
  using AggGroupToNode = Map<uint32_t, int32_t>;

  // the join_group_mapping of the join tid to the agg tid, for each node
  using JoinGroups = Vector<std::pair<uint32_t, uint32_t>>;
  using JoinGroupsPerNode = Vector<JoinGroups>;

  // the left side record mappings
  Handle<JoinTIDToNode> leftToNode;

  // the right side record mappings
  Handle<JoinTIDToNode> rightToNode;

  // tells us what join groups are joined
  Handle<JoinGroupsPerNode> joinGroupsPerNode;

  // the aggregation group mappings
  Handle<AggGroupToNode> aggToNode;
};

}