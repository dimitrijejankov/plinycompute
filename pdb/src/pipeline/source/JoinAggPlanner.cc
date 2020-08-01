#include <GreedyPlanner.h>
#include <thread>
#include "JoinAggPlanner.h"
#include "GeneticAggGroupPlanner.h"

pdb::JoinAggPlanner::JoinAggPlanner(const pdb::PDBAnonymousPageSetPtr &joinAggPageSet,
                                    uint32_t numNodes,
                                    uint32_t numThreads,
                                    const PDBPageHandle &pageToStore) : numNodes(numNodes),
                                                                        numThreads(numThreads) {
  // get the input page
  this->inputPage = joinAggPageSet->getNextPage(0);
  if(this->inputPage == nullptr) {
    throw runtime_error("There are no keys to do planning...");
  }
  this->inputPage->repin();

  // page to store
  this->pageToStore = pageToStore;

  // we have not executed anything
  this->num_finished = 0;

  // grab the copy of the aggGroups object
  auto *record = (Record<TIDIndexMap> *) inputPage->getBytes();
  aggGroups = record->getRootObject();
}

void pdb::JoinAggPlanner::doPlanning() {

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // we need this for the planner
  std::vector<char> lhsRecordPositions;
  lhsRecordPositions.reserve(100);

  std::vector<char> rhsRecordPositions;
  rhsRecordPositions.reserve(100);

  std::vector<std::vector<int32_t>> aggregationGroups;
  aggregationGroups.resize(this->aggGroups->size());

  // figure the number of join groups berforehand
  auto numJoinGroups = 0;
  for (auto it = this->aggGroups->begin(); it != this->aggGroups->end(); ++it) { numJoinGroups += (*it).value.size(); }

  // resize the join groups the the right size
  // join groups belonging to the same aggregation group will be grouped together
  std::vector<PipJoinAggPlanResult::JoinedRecord> joinGroups;
  joinGroups.resize(numJoinGroups);

  // I use these to keep track of what
  int32_t currentJoinTID = 0;
  int32_t currentAggGroup = 0;
  for (auto it = this->aggGroups->begin(); it != this->aggGroups->end(); ++it) {

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

      // store the join group
      joinGroups[currentJoinTID] = { leftTID, rightTID, aggTID};

      // resize if necessary
      if (lhsRecordPositions.size() <= (leftTID + 1)) {
        lhsRecordPositions.resize((leftTID + 1));
      }

      // resize if necessary
      if (rhsRecordPositions.size() <= (rightTID + 1)) {
        rhsRecordPositions.resize((rightTID + 1));
      }

      // set the tids
      lhsRecordPositions[leftTID] = leftTIDNode;
      rhsRecordPositions[rightTID] = rightTIDNode;

      // set the tid to the group
      aggregationGroups[currentAggGroup].emplace_back(currentJoinTID);

      // go to the next join TID
      currentJoinTID++;
    }

    // we finished processing an aggregation group
    currentAggGroup++;
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Prep run : " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "[ns]" << '\n';

  // dod the full planning
  doFullPlanning(lhsRecordPositions, rhsRecordPositions, aggregationGroups, joinGroups);

  std::cout << "Finished Planning\n";
}

void pdb::JoinAggPlanner::doFullPlanning(const std::vector<char> &lhsRecordPositions,
                                         const std::vector<char> &rhsRecordPositions,
                                         const std::vector<std::vector<int32_t>> &aggregationGroups,
                                         const std::vector<PipJoinAggPlanResult::JoinedRecord> &joinGroups) {

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  pdb::GeneticAggGroupPlanner::costs_t c{};
  c.agg_cost = 1;
  c.join_cost = 1;
  c.join_rec_size = 1;
  c.send_coef = 1;
  c.rhs_input_rec_size = 1;
  c.lhs_input_rec_size = 1;
  c.aggregation_rec_size = 1;

  // init the planner run the agg only planner
  pdb::GeneticAggGroupPlanner planner(10,
                                      numNodes,
                                      aggregationGroups.size(),
                                      joinGroups.size(),
                                      lhsRecordPositions.size(),
                                      rhsRecordPositions.size(),
                                      c,
                                      lhsRecordPositions,
                                      rhsRecordPositions,
                                      aggregationGroups,
                                      joinGroups);

  // run for a number of iterations
  planner.run(400);

  // get the result of the planning
  auto result = planner.get_result();

  // repin the page
  pageToStore->repin();
  UseTemporaryAllocationBlock blk{pageToStore->getBytes(), pageToStore->getSize()};

  // make the plan result object
  Handle<PipJoinAggPlanResult> planResult = pdb::makeObject<PipJoinAggPlanResult>(numNodes);

  // set the number of aggregation groups
  planResult->numAggGroups = this->aggGroups->size();

  // go through the map and do two things
  // assign aggregation groups to nodes
  for (auto it = this->aggGroups->begin(); it != this->aggGroups->end(); ++it) {

    /// 0. Round robing the aggregation groups

    // the aggregation tid
    auto &aggTID = (*it).key;

    // get the current node
    auto assignedNode = result.agg_group_assignments[aggTID];

    // assign the aggregation group to the node
    (*planResult->aggToNode)[aggTID] = assignedNode;
  }

  // keeps track of what the last aggregation group was
  vector<int32_t> lastAggGroup(numNodes, -1);

  // go through each join group
  for(auto jg = 0; jg < joinGroups.size(); ++jg) {

    for(auto node = 0; node < numNodes; ++node) {
      if (result.join_groups_to_node[jg * numNodes + node]) {

        /// 1.0 Store the left side
        {
          // make sure we have it
          if ((*planResult->leftToNode).count(joinGroups[jg].lhsTID) == 0) {
            (*planResult->leftToNode)[joinGroups[jg].lhsTID] = Vector<bool>(numNodes, numNodes);
            (*planResult->leftToNode)[joinGroups[jg].lhsTID].fill(false);
          }

          // grab the vector for the key tid
          (*planResult->leftToNode)[joinGroups[jg].lhsTID][node] = true;
        }

        /// 1.1 Store the right side
        {
          // make sure we have it
          if ((*planResult->rightToNode).count(joinGroups[jg].rhsTID) == 0) {
            (*planResult->rightToNode)[joinGroups[jg].rhsTID] = Vector<bool>(numNodes, numNodes);
            (*planResult->rightToNode)[joinGroups[jg].rhsTID].fill(false);
          }

          // grab the vector for the key tid
          (*planResult->rightToNode)[joinGroups[jg].rhsTID][node] = true;
        }

        /// 1.2 Store the join group
        (*planResult->joinGroupsPerNode)[node].push_back({joinGroups[jg].lhsTID, joinGroups[jg].rhsTID, joinGroups[jg].aggTID});

        /// 1.3 Update the num of aggregation groups per node if this is a new aggregation group
        (*planResult->numAggGroupsPerNode)[node] += joinGroups[jg].aggTID != lastAggGroup[node];
        lastAggGroup[node] = joinGroups[jg].aggTID;
      }
    }
  }

  // mark that we are using the whole
  selectedAlgorithm = AlgorithmID::FULL;

  // set the main record of the page
  getRecord(planResult);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  full_first_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

bool pdb::JoinAggPlanner::isLocalAggregation() {
  return selectedAlgorithm == AlgorithmID::AGG_FIRST_ONLY;
}

void pdb::JoinAggPlanner::print(const Handle<PipJoinAggPlanResult> &planResult) {

  for (auto it = planResult->leftToNode->begin(); it != planResult->leftToNode->end(); ++it) {

    std::cout << "Left TID " << (*it).key << " goes to:\n";
    Vector<bool> &nodes = (*it).value;
    for (int i = 0; i < nodes.size(); ++i) {
      if (nodes[i]) {
        std::cout << "\tNode " << i << "\n";
      }
    }
  }

  std::cout << "\n\n";

  for (auto it = planResult->rightToNode->begin(); it != planResult->rightToNode->end(); ++it) {

    std::cout << "Right TID " << (*it).key << " goes to:\n";
    Vector<bool> &nodes = (*it).value;
    for (int i = 0; i < nodes.size(); ++i) {
      if (nodes[i]) {
        std::cout << "\tNode " << i << "\n";
      }
    }
  }

  std::cout << "\n\n";

  for (auto it = planResult->aggToNode->begin(); it != planResult->aggToNode->end(); ++it) {
    std::cout << "Aggregation Group" << (*it).key << " goes to " << (*it).value << "\n";
  }
  std::cout << "\n\n";
}