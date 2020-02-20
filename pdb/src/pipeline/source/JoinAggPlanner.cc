#include "JoinAggPlanner.h"
#include "GeneticAlgorithmPlanner.h"

pdb::JoinAggPlanner::JoinAggPlanner(const pdb::PDBAnonymousPageSetPtr &joinAggPageSet,
                                    uint32_t numNodes,
                                    uint32_t numThreads,
                                    const PDBPageHandle& pageToStore) : numNodes(numNodes),
                                                                        numThreads(numThreads) {


  // get the input page
  this->inputPage = joinAggPageSet->getNextPage(0);
  this->inputPage->repin();

  // page to store
  this->pageToStore = pageToStore;

  // grab the copy of the aggGroups object
  auto* record = (Record<TIDIndexMap>*) inputPage->getBytes();
  aggGroups = record->getRootObject();
}

void pdb::JoinAggPlanner::doPlanning() {

  // repin the page
  pageToStore->repin();
  UseTemporaryAllocationBlock blk{pageToStore->getBytes(), pageToStore->getSize()};

  // make the plan result object
  Handle<PipJoinAggPlanResult> planResult = pdb::makeObject<PipJoinAggPlanResult>(numNodes);

  // we need this for the planner
  std::vector<std::vector<bool>> left_tids;
  std::vector<std::vector<bool>> right_tids;

  // there is going to be
  std::vector<std::vector<std::pair<int32_t, int32_t>>> aggregation_groups;
  aggregation_groups.resize(this->aggGroups->size());

  //
  for(auto it = this->aggGroups->begin(); it != this->aggGroups->end(); ++it) {

    /// 0. Round robing the aggregation groups

    // assign the
    TIDVector &joinedTIDs = (*it).value;
    auto &aggTID = (*it).key;

    std::cout << "Aggregation group : " << aggTID << '\n';

    // the join pairs
    std::vector<std::pair<int32_t, int32_t>> aggregation_group(joinedTIDs.size());

    for(size_t i = 0; i < joinedTIDs.size(); ++i) {

      // get the left tid
      auto leftTID = joinedTIDs[i].first.first;
      auto leftTIDNode = joinedTIDs[i].first.second;

      // get the right tid
      auto rightTID = joinedTIDs[i].second.first;
      auto rightTIDNode = joinedTIDs[i].second.second;

      std::cout << "Join group ( " << leftTID << ", " << rightTID << ")\n";

      // resize if necessary
      if(left_tids.size() <= leftTID) {
        left_tids.resize(leftTID + 1);
      }

      // resize if necessary
      if(right_tids.size() <= rightTID) {
        right_tids.resize(rightTID + 1);
      }

      // resize the tid if necessary
      if(left_tids[leftTID].empty()) {
        left_tids[leftTID].resize(numNodes);
      }

      // resize the tid if necessary
      if(right_tids[rightTID].empty()) {
        right_tids[rightTID].resize(numNodes);
      }

      // set the tids
      left_tids[leftTID][leftTIDNode] = true;
      right_tids[rightTID][rightTIDNode] = true;

      // set the tid to the group
      aggregation_group[i] = std::make_pair(leftTID, rightTID);
    }

    // store the aggregation group
    aggregation_groups[aggTID] = std::move(aggregation_group);
  }

  std::cout << '\n';

  auto numLeftRecords = left_tids.size();
  auto numRightRecords = right_tids.size();
  auto numAggregationRecords = aggregation_groups.size();

  pdb::GeneticAlgorithmPlanner planner(numLeftRecords,
                                       numRightRecords,
                                       numNodes,
                                       numAggregationRecords,
                                       1, // need to get this parameter
                                       1, // need to get this parameter
                                       std::move(left_tids),
                                       std::move(right_tids),
                                       std::move(aggregation_groups),
                                       40);


  // run for a number of iterations
  planner.run(50);

  // get the result of the planning
  auto result = planner.getResult();

  // go through the map and do two things
  // assign aggregation groups to nodes
  for(auto it = this->aggGroups->begin(); it != this->aggGroups->end(); ++it) {

    /// 0. Round robing the aggregation groups

    // the aggregation tid
    auto &aggTID = (*it).key;

    // get the current node
    auto assignedNode = result[aggTID];

    // assign the aggregation group to the node
    (*planResult->aggToNode)[aggTID] = assignedNode;

    // assign the
    TIDVector &joinedTIDs = (*it).value;

    // go through each joined key that makes up this and store what node we need to send it
    for(size_t i = 0; i < joinedTIDs.size(); ++i) {

      /// 1.0 Store the left side
      {
        // make sure we have it
        if((*planResult->leftToNode).count(joinedTIDs[i].first.first) == 0) {
          (*planResult->leftToNode)[joinedTIDs[i].first.first] = Vector<bool>(numNodes, numNodes);
          (*planResult->leftToNode)[joinedTIDs[i].first.first].fill(false);
        }

        // grab the vector for the key tid
        (*planResult->leftToNode)[joinedTIDs[i].first.first][assignedNode] = true;
      }

      /// 1.1 Store the right side
      {
        // make sure we have it
        if((*planResult->rightToNode).count(joinedTIDs[i].second.first) == 0) {
          (*planResult->rightToNode)[joinedTIDs[i].second.first] = Vector<bool>(numNodes, numNodes);
          (*planResult->rightToNode)[joinedTIDs[i].second.first].fill(false);
        }

        // grab the vector for the key tid
        (*planResult->rightToNode)[joinedTIDs[i].second.first][assignedNode] = true;
      }

      /// 1.2 Store the join group
      {
        (*planResult->joinGroupsPerNode)[assignedNode].push_back(std::make_pair(joinedTIDs[i].first.first, joinedTIDs[i].second.first));
      }
    }
  }

  // set the main record of the page
  getRecord(planResult);

  // print the planning result
  print(planResult);
}

void pdb::JoinAggPlanner::print(const Handle<PipJoinAggPlanResult> &planResult) {

  for(auto it = planResult->leftToNode->begin(); it != planResult->leftToNode->end(); ++it) {

    std::cout << "Left TID " << (*it).key << " goes to:\n";
    Vector<bool> &nodes = (*it).value;
    for(int i = 0; i < nodes.size(); ++i) {
      if(nodes[i]) {
        std::cout << "\tNode " << i << "\n";
      }
    }
  }

  std::cout << "\n\n";

  for(auto it = planResult->rightToNode->begin(); it != planResult->rightToNode->end(); ++it) {

    std::cout << "Right TID " << (*it).key << " goes to:\n";
    Vector<bool> &nodes = (*it).value;
    for(int i = 0; i < nodes.size(); ++i) {
      if(nodes[i]) {
        std::cout << "\tNode " << i << "\n";
      }
    }
  }

  std::cout << "\n\n";

  for(auto it = planResult->aggToNode->begin(); it != planResult->aggToNode->end(); ++it) {
    std::cout << "Aggregation Group" << (*it).key << " goes to " << (*it).value <<"\n";
  }
}

