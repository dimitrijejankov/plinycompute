#pragma once


// PRELOAD %PipJoinAggPlanResult%

namespace pdb {

// encapsulates a request to add data to a set in storage
class PipJoinAggPlanResult : public Object {

public:

  PipJoinAggPlanResult() {

    // init stuff
    this->leftToNode = pdb::makeObject<PipJoinAggPlanResult::JoinTIDToNode>();
    this->rightToNode = pdb::makeObject<PipJoinAggPlanResult::JoinTIDToNode>();
    this->aggToNode = pdb::makeObject<PipJoinAggPlanResult::AggGroupToNode >();
  };


  ~PipJoinAggPlanResult() = default;

  ENABLE_DEEP_COPY

  // join to node map
  using JoinTIDToNode = Map<uint32_t, Vector<bool>>;

  // aggregation group to node map
  using AggGroupToNode = Map<uint32_t, int32_t>;

  // the left side record mappings
  Handle<JoinTIDToNode> leftToNode;

  // the right side record mappings
  Handle<JoinTIDToNode> rightToNode;

  // the aggregation group mappings
  Handle<AggGroupToNode> aggToNode;
};

}