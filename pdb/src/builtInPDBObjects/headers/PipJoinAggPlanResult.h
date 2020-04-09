#pragma once

#include <Object.h>
#include <PDBString.h>
#include <PDBMap.h>
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

  struct JoinedRecord {
    uint32_t lhsTID;
    uint32_t rhsTID;
    uint32_t aggTID;
  };

  // join to node map
  using JoinTIDToNode = Map<uint32_t, Vector<bool>>;

  // aggregation group to node map1
  using AggGroupToNode = Map<uint32_t, int32_t>;

  // join groups have lhsTID, rhsTID and the aggregation group TID
  using JoinGroups = Vector<JoinedRecord>;
  using JoinGroupsPerNode = Vector<JoinGroups>;

  // the left side record mappings
  Handle<JoinTIDToNode> leftToNode;

  // the right side record mappings
  Handle<JoinTIDToNode> rightToNode;

  // tells us what join groups are joined
  Handle<JoinGroupsPerNode> joinGroupsPerNode;

  // the aggregation group mappings
  Handle<AggGroupToNode> aggToNode;

  // is it local only
  bool isLocalAggregation = false;

  // how many aggregation groups are there
  int32_t numAggGroups = 0;
};

}