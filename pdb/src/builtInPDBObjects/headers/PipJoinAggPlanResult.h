#pragma once


// PRELOAD %PipJoinAggPlanResult%

namespace pdb {

// encapsulates a request to add data to a set in storage
class PipJoinAggPlanResult : public Object {

public:

  PipJoinAggPlanResult() = default;

  explicit PipJoinAggPlanResult(int32_t numThreads) {

    // init stuff
    this->leftToNode = pdb::makeObject<PipJoinAggPlanResult::JoinTIDToNode>();
    this->rightToNode = pdb::makeObject<PipJoinAggPlanResult::JoinTIDToNode>();
    this->aggToNode = pdb::makeObject<PipJoinAggPlanResult::AggGroupToNode >();
    this->leftTidToAggGroup = pdb::makeObject<PipJoinAggPlanResult::TIDtoGroupTIDForNodes>(numThreads, numThreads);
    this->rightTidToAggGroup = pdb::makeObject<PipJoinAggPlanResult::TIDtoGroupTIDForNodes>(numThreads, numThreads);
  };


  ~PipJoinAggPlanResult() = default;

  ENABLE_DEEP_COPY

  // join to node map
  using JoinTIDToNode = Map<uint32_t, Vector<bool>>;

  // aggregation group to node map1
  using AggGroupToNode = Map<uint32_t, int32_t>;

  // the mapping of the join tid to the agg tid, for each node
  using TIDtoGroupTIDForNode = Map<uint32_t, uint32_t>;
  using TIDtoGroupTIDForNodes = Vector<TIDtoGroupTIDForNode>;

  // the left side record mappings
  Handle<JoinTIDToNode> leftToNode;

  // the right side record mappings
  Handle<JoinTIDToNode> rightToNode;

  // the mapping of the left tid to the agg group tid, for each node
  Handle<TIDtoGroupTIDForNodes> leftTidToAggGroup;

  // the mapping of the right tid to the agg group tid, for each node
  Handle<TIDtoGroupTIDForNodes> rightTidToAggGroup;

  // the aggregation group mappings
  Handle<AggGroupToNode> aggToNode;
};

}