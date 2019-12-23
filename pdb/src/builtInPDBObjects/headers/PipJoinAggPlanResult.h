#pragma once


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

  void merge(std::unordered_map<uint32_t, uint32_t> &m, uint32_t from, uint32_t to) {

    // merge all the from to the to
    for(auto &v : m) {
      if(v.second == from) {
        v.second = to;
      }
    }
  }

  void generateHashes(int currentNode, std::unordered_map<uint32_t, uint32_t> &left, std::unordered_map<uint32_t, uint32_t> &right) {

    uint32_t nextID = 1;
    for(int i = 0; i < joinGroupsPerNode->size(); ++i) {

      // get the tids that are joined
      auto leftTID = (*joinGroupsPerNode)[currentNode][i].first;
      auto rightTID = (*joinGroupsPerNode)[currentNode][i].second;

      // if it is zero assign it something
      if (left[leftTID] == 0 && right[rightTID] == 0) {
        left[leftTID] = nextID;
        right[rightTID] = nextID++;
        continue;
      }

      // if both of them are set merge them
      if (left[leftTID] != 0 && right[rightTID] != 0) {

        // we merge the left to right
        uint32_t from = left[leftTID];
        uint32_t to = right[rightTID];

        // we are merging both sides
        merge(left, from, to);
        merge(right, from, to);
      }

      // if the first one is zero that means the other is not
      if (left[leftTID] == 0) {
        left[leftTID] = right[rightTID];
      }

      // if the second one is zer that means the other is not
      if (right[rightTID] == 0) {
        right[rightTID] = left[leftTID];
      }
    }
  }

  ENABLE_DEEP_COPY

  // join to node map
  using JoinTIDToNode = Map<uint32_t, Vector<bool>>;

  // aggregation group to node map1
  using AggGroupToNode = Map<uint32_t, int32_t>;

  // the mapping of the join tid to the agg tid, for each node
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