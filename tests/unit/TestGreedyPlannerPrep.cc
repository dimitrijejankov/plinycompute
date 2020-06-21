#include <iostream>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <PipJoinAggPlanResult.h>
#include <JoinAggPlanner.h>
#include <PDBVector.h>
#include <GreedyPlanner.h>

// parameters
uint64_t num_row_ids_a = 80;
uint64_t num_col_ids_a = 80;
uint64_t num_row_ids_b = 80;
uint64_t num_col_ids_b = 80;
uint64_t num_nodes = 10;

struct record_t {

  int32_t rowID;
  int32_t colID;
  int32_t node;
};

bool operator==(const record_t &lhs, const record_t &rhs) {
  return lhs.rowID == rhs.rowID && lhs.colID == rhs.colID;
}

inline std::size_t hash_combine(std::size_t h1, std::size_t h2) {
  h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
  return h1;
}

struct record_hasher_t {
  std::size_t operator()(record_t const &s) const noexcept {
    std::size_t h1 = std::hash<int32_t>{}(s.rowID);
    std::size_t h2 = std::hash<int32_t>{}(s.colID);
    return hash_combine(h1, h2)
  }
};

struct join_group_t {
  int32_t lhs;
  int32_t rhs;
  int32_t agg = -1;
};

bool operator==(const join_group_t &lhs, const join_group_t &rhs) {
  return lhs.lhs == rhs.lhs && lhs.rhs == rhs.rhs;
}

struct join_group_hasher_t {
  std::size_t operator()(join_group_t const &s) const noexcept {
    std::size_t h1 = std::hash<int32_t>{}(s.lhs);
    std::size_t h2 = std::hash<int32_t>{}(s.rhs);
    return h1 ^ (h2 << 1);
  }
};


// we use this to deduplicate the join groups
struct join_group_hasher {
  size_t operator()(const pair<int32_t, int32_t> &p) const {
    auto hash1 = hash<int32_t>{}(p.first);
    auto hash2 = hash<int32_t>{}(p.second);
    return hash1 ^ hash2;
  }
};

// the first integer is the join key value identifier, the second value is the node it comes from
using TIDType = std::pair<uint32_t, int32_t>;

using TIDVector = pdb::Vector<std::pair<TIDType, TIDType>>;

using TIDIndexMap = pdb::Map<uint32_t, TIDVector>;

void initTIDMap(pdb::Handle<TIDIndexMap> &aggGroups, void *block, std::size_t numBytes) {

  pdb::makeObjectAllocatorBlock(block, numBytes, true);

  // the lhs records
  std::vector<record_t> lhs_records;
  for (auto row_id_a = 0; row_id_a < num_row_ids_a; ++row_id_a) {
    for (auto col_id_a = 0; col_id_a < num_col_ids_a; ++col_id_a) {
      int32_t node = ((int32_t) rand() % num_nodes);
      lhs_records.push_back({.rowID = row_id_a, .colID = col_id_a, .node = node});
    }
  }

  // the rhs records
  std::vector<record_t> rhs_records;
  for (auto row_id_b = 0; row_id_b < num_row_ids_b; ++row_id_b) {
    for (auto col_id_b = 0; col_id_b < num_col_ids_b; ++col_id_b) {
      int32_t node = ((int32_t) rand() % num_nodes);
      rhs_records.push_back({.rowID = row_id_b, .colID = col_id_b, .node = node});
    }
  }

  // join groups
  std::unordered_map<join_group_t, int32_t, join_group_hasher_t> join_groups;
  std::unordered_map<record_t, std::vector<int32_t>, record_hasher_t> agg_groups;
  for (auto lhs = 0; lhs < lhs_records.size(); ++lhs) {
    for (auto rhs = 0; rhs < rhs_records.size(); ++rhs) {

      // check if they match
      if (lhs_records[lhs].colID == rhs_records[rhs].rowID) {

        // make the key
        join_group_t key{.lhs = lhs, .rhs = rhs};

        // try to find the
        auto it = join_groups.find(key);
        int32_t id = 0;

        // try to find it, if not assign it
        if (it == join_groups.end()) { id = (join_groups[key] = join_groups.size()); } else { id = it->second; }

        // add it to the aggregation group
        agg_groups[record_t{.rowID = lhs_records[lhs].rowID, .colID = rhs_records[rhs].colID}].push_back(id);
      }
    }
  }

  std::vector<pdb::PipJoinAggPlanResult::JoinedRecord> join_groups_vec;
  join_groups_vec.resize(join_groups.size());
  for (auto &jg : join_groups) {
    join_groups_vec[jg.second] = pdb::PipJoinAggPlanResult::JoinedRecord{ static_cast<uint32_t>(jg.first.lhs),
                                                                          static_cast<uint32_t>(jg.first.rhs),
                                                                          0u};
  }

  std::vector<std::vector<int32_t>> aggregation_groups;
  aggregation_groups.resize(agg_groups.size());
  int32_t i = 0;
  for (auto &agg : agg_groups) {
    for (auto &jg : agg.second) {
      aggregation_groups[i].emplace_back(jg);
      join_groups_vec[jg].aggTID = i;
    }
  }

  // make the object
  aggGroups = pdb::makeObject<TIDIndexMap>();

  for(int32_t jg = 0; jg < join_groups_vec.size(); jg++) {

    auto leftTID = join_groups_vec[jg].lhsTID;
    auto rightTID = join_groups_vec[jg].rhsTID;
    auto leftNode = lhs_records[join_groups_vec[jg].lhsTID].node;
    auto rightNode = rhs_records[join_groups_vec[jg].rhsTID].node;

    if (aggGroups->count(join_groups_vec[jg].aggTID) == 0) {

      // copy and add to hash map
      TIDVector &temp = (*aggGroups)[join_groups_vec[jg].aggTID];
      temp = TIDVector();
      temp.push_back({{leftTID, leftNode}, {rightTID, rightNode}});
    }
    else {

      // copy and add to hash map
      TIDVector &temp = (*aggGroups)[join_groups_vec[jg].aggTID];
      temp.push_back({{leftTID, leftNode}, {rightTID, rightNode}});
    }
  }

};

int main() {

  // the the map
  pdb::Handle<TIDIndexMap> aggGroups;
  void* buffer = malloc(128 * 1024 * 1024);
  initTIDMap(aggGroups, buffer, 128 * 1024 * 1024);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // we need this for the planner
  std::vector<char> lhsRecordPositions;
  lhsRecordPositions.reserve(100 * num_nodes);

  std::vector<char> rhsRecordPositions;
  rhsRecordPositions.reserve(100 * num_nodes);

  std::vector<std::vector<int32_t>> aggregationGroups;
  aggregationGroups.resize(aggGroups->size());

  std::vector<pdb::PipJoinAggPlanResult::JoinedRecord> joinGroups;
  joinGroups.reserve(4000);

  //
  int32_t currentJoinTID = 0;
  std::unordered_map<std::pair<int32_t, int32_t>, int32_t, join_group_hasher> deduplicatedGroups;

  auto currentAggGroup = 0;
  for (auto it = aggGroups->begin(); it != aggGroups->end(); ++it) {

    /// 0. Round robing the aggregation groups

    // assign the
    TIDVector &joinedTIDs = (*it).value;
    auto &aggTID = (*it).key;

    // the join pairs
    for (size_t i = 0; i < joinedTIDs.size(); ++i) {

      // get the left tid
      auto leftTID = joinedTIDs[i].first.first;
      auto leftTIDNode = joinedTIDs[i].first.second;

      // get the right tid
      auto rightTID = joinedTIDs[i].second.first;
      auto rightTIDNode = joinedTIDs[i].second.second;

      // the join group
      auto jg = std::make_pair(leftTID, rightTID);
      auto jg_it = deduplicatedGroups.find(jg);

      // if we don't have a id assigned to the join group assign one
      if (jg_it == deduplicatedGroups.end()) {

        // resize the tid
        joinGroups.resize(currentJoinTID + 1);

        // store the join group
        joinGroups[currentJoinTID] = { jg.first, jg.second, aggTID };
        deduplicatedGroups[jg] = currentJoinTID;
        currentJoinTID++;
      }

      // the tid
      int32_t jg_tid = deduplicatedGroups[jg];

      // resize if necessary
      if (lhsRecordPositions.size() <= ((leftTID + 1) * num_nodes)) {
        lhsRecordPositions.resize(((leftTID + 1) * num_nodes));
      }

      // resize if necessary
      if (rhsRecordPositions.size() <= ((rightTID + 1) * num_nodes)) {
        rhsRecordPositions.resize(((rightTID + 1) * num_nodes));
      }

      // set the tids
      lhsRecordPositions[leftTID * num_nodes + leftTIDNode] = true;
      rhsRecordPositions[rightTID * num_nodes + rightTIDNode] = true;

      // set the tid to the group
      aggregationGroups[currentAggGroup].emplace_back(jg_tid);
    }

    // we finished processing an aggregation group
    currentAggGroup++;
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Stuff blabla " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "[ns]" << '\n';

  // set the costs
  pdb::GreedyPlanner::costs_t c{};
  c.agg_cost = 1;
  c.join_cost = 1;
  c.join_rec_size = 1;
  c.send_coef = 1;
  c.rhs_input_rec_size = 1;
  c.lhs_input_rec_size = 1;
  c.aggregation_rec_size = 1;

  pdb::GreedyPlanner planner(num_nodes,
                             c,
                             lhsRecordPositions,
                             rhsRecordPositions,
                             aggregationGroups,
                             joinGroups);

  std::chrono::steady_clock::time_point begine = std::chrono::steady_clock::now();

  planner.run();
  planner.print();
  std::chrono::steady_clock::time_point ende = std::chrono::steady_clock::now();
  std::cout << "Planner run for " << std::chrono::duration_cast<std::chrono::nanoseconds>(ende - begin).count() << "[ns]" << '\n';

  planner.print();

  return 0;
}