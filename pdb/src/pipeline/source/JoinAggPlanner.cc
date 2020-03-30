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
  this->inputPage->repin();

  // page to store
  this->pageToStore = pageToStore;

  // we have not executed anything
  this->spinOnMe = 0;

  // grab the copy of the aggGroups object
  auto *record = (Record<TIDIndexMap> *) inputPage->getBytes();
  aggGroups = record->getRootObject();
}

void pdb::JoinAggPlanner::doPlanning() {

  // we need this for the planner
  std::vector<char> lhsRecordPositions;
  lhsRecordPositions.reserve(100 * numNodes);

  std::vector<char> rhsRecordPositions;
  rhsRecordPositions.reserve(100 * numNodes);

  std::vector<std::vector<int32_t>> aggregationGroups;
  aggregationGroups.resize(this->aggGroups->size());

  std::vector<std::pair<int32_t, int32_t>> joinGroups;
  joinGroups.resize(1000);

  //
  int32_t currentJoinTID = 0;
  unordered_map<pair<int32_t, int32_t>, int32_t, join_group_hasher> deduplicatedGroups;

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

      // the join group
      auto jg = std::make_pair(leftTID, rightTID);
      auto jg_it = deduplicatedGroups.find(jg);

      // if we don't have a id assigned to the join group assign one
      if (jg_it != deduplicatedGroups.end()) {
        deduplicatedGroups[jg] = currentJoinTID++;
      }

      // the tid
      int32_t jg_tid = deduplicatedGroups[jg];

      // resize if necessary
      if (lhsRecordPositions.size() <= ((leftTID + 1) * numNodes)) {
        lhsRecordPositions.resize(((leftTID + 1) * numNodes));
      }

      // resize if necessary
      if (rhsRecordPositions.size() <= ((rightTID + 1) * numNodes)) {
        rhsRecordPositions.resize(((rightTID + 1) * numNodes));
      }

      // set the tids
      lhsRecordPositions[leftTID * numNodes + leftTIDNode] = true;
      rhsRecordPositions[rightTID * numNodes + rightTIDNode] = true;

      // set the tid to the group
      aggregationGroups[i].emplace_back(jg_tid);
    }
  }

  // do the planning with just aggregation
  std::thread th_do_agg_first([&](){
    doAggFirstPlanning(lhsRecordPositions, rhsRecordPositions, aggregationGroups, joinGroups);
  });

  // do the planning with just join
  std::thread th_do_join_first([&](){
    doJoinFirstPlanning(lhsRecordPositions, rhsRecordPositions, aggregationGroups, joinGroups);
  });

  // do the planning with just join
  std::thread th_do_full([&](){
    doFullPlanning(lhsRecordPositions, rhsRecordPositions, aggregationGroups, joinGroups);
  });

  // wait for this to finish
  th_do_agg_first.join();
  th_do_join_first.join();
  th_do_full.join();

  std::cout << "Finished Planning\n";
}

void pdb::JoinAggPlanner::doAggFirstPlanning(const std::vector<char> &lhsRecordPositions,
                                             const std::vector<char> &rhsRecordPositions,
                                             const std::vector<std::vector<int32_t>> &aggregationGroups,
                                             const std::vector<std::pair<int32_t, int32_t>> &joinGroups) {

  pdb::GreedyPlanner::costs_t c{};
  c.agg_cost = 1;
  c.join_cost = 1;
  c.join_rec_size = 1;
  c.send_coef = 1;
  c.rhs_input_rec_size = 1;
  c.lhs_input_rec_size = 1;
  c.aggregation_rec_size = 1;

  // init the planner run the agg only planner
  pdb::GreedyPlanner planner(numNodes, c, lhsRecordPositions, rhsRecordPositions, aggregationGroups, joinGroups);

  // run for a number of iterations
  planner.run_agg_first_only();

  // get the result of the planning
  auto result = planner.get_agg_result();

  // update the planning costs
  planning_costs[(int32_t) AlgorithmID::AGG_FIRST_ONLY] = planner.getCost();

  // spin while all the other planners don't finish
  num_finished++;
  while (num_finished != 3) {}

  // the algorithm we selected
  auto choice = selectAlgorithm();

  // if the choice aggregation algorithm then do the rest of the planning
  if (choice == AlgorithmID::AGG_FIRST_ONLY) {

    // repin the page
    pageToStore->repin();
    UseTemporaryAllocationBlock blk{pageToStore->getBytes(), pageToStore->getSize()};

    // make the plan result object
    Handle<PipJoinAggPlanResult> planResult = pdb::makeObject<PipJoinAggPlanResult>(numNodes);

    // go through the map and do two things
    // assign aggregation groups to nodes
    for (auto it = this->aggGroups->begin(); it != this->aggGroups->end(); ++it) {

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
      for (size_t i = 0; i < joinedTIDs.size(); ++i) {

        /// 1.0 Store the left side
        {
          // make sure we have it
          if ((*planResult->leftToNode).count(joinedTIDs[i].first.first) == 0) {
            (*planResult->leftToNode)[joinedTIDs[i].first.first] = Vector<bool>(numNodes, numNodes);
            (*planResult->leftToNode)[joinedTIDs[i].first.first].fill(false);
          }

          // grab the vector for the key tid
          (*planResult->leftToNode)[joinedTIDs[i].first.first][assignedNode] = true;
        }

        /// 1.1 Store the right side
        {
          // make sure we have it
          if ((*planResult->rightToNode).count(joinedTIDs[i].second.first) == 0) {
            (*planResult->rightToNode)[joinedTIDs[i].second.first] = Vector<bool>(numNodes, numNodes);
            (*planResult->rightToNode)[joinedTIDs[i].second.first].fill(false);
          }

          // grab the vector for the key tid
          (*planResult->rightToNode)[joinedTIDs[i].second.first][assignedNode] = true;
        }

        /// 1.2 Store the join group
        {
          (*planResult->joinGroupsPerNode)[assignedNode].push_back(
              std::make_pair(joinedTIDs[i].first.first, joinedTIDs[i].second.first));
        }
      }
    }

    // set the main record of the page
    getRecord(planResult);

    // store the choice
    selectedAlgorithm = AlgorithmID::AGG_FIRST_ONLY;
  }
}

void pdb::JoinAggPlanner::doJoinFirstPlanning(const std::vector<char> &lhsRecordPositions,
                                              const std::vector<char> &rhsRecordPositions,
                                              const std::vector<std::vector<int32_t>> &aggregationGroups,
                                              const std::vector<std::pair<int32_t, int32_t>> &joinGroups) {

  pdb::GreedyPlanner::costs_t c{};
  c.agg_cost = 1;
  c.join_cost = 1;
  c.join_rec_size = 1;
  c.send_coef = 1;
  c.rhs_input_rec_size = 1;
  c.lhs_input_rec_size = 1;
  c.aggregation_rec_size = 1;

  // init the planner run the agg only planner
  pdb::GreedyPlanner planner(numNodes, c, lhsRecordPositions, rhsRecordPositions, aggregationGroups, joinGroups);

  // run for a number of iterations
  planner.run_join_first_only();

  // get the result of the planning
  auto result = planner.get_result();

  // update the planning costs
  planning_costs[(int32_t) AlgorithmID::JOIN_FIRST_ONLY] = planner.getCost();

  // spin while all the other planners don't finish
  num_finished++;
  while (num_finished != 3) {}

  // the algorithm we selected
  auto choice = selectAlgorithm();

  //
  if (choice == AlgorithmID::JOIN_FIRST_ONLY) {

    // repin the page
    pageToStore->repin();
    UseTemporaryAllocationBlock blk{pageToStore->getBytes(), pageToStore->getSize()};

    // make the plan result object
    Handle<PipJoinAggPlanResult> planResult = pdb::makeObject<PipJoinAggPlanResult>(numNodes);

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

    // go through each join group
    for(auto jg = 0; jg < joinGroups.size(); ++jg) {

      for(auto node = 0; node < numNodes; ++node) {
        if (result.join_groups_to_node[jg * numNodes + node]) {

          /// 1.0 Store the left side
          {
            // make sure we have it
            if ((*planResult->leftToNode).count(joinGroups[jg].first) == 0) {
              (*planResult->leftToNode)[joinGroups[jg].first] = Vector<bool>(numNodes, numNodes);
              (*planResult->leftToNode)[joinGroups[jg].first].fill(false);
            }

            // grab the vector for the key tid
            (*planResult->leftToNode)[joinGroups[jg].first][node] = true;
          }

          /// 1.1 Store the right side
          {
            // make sure we have it
            if ((*planResult->rightToNode).count(joinGroups[jg].second) == 0) {
              (*planResult->rightToNode)[joinGroups[jg].second] = Vector<bool>(numNodes, numNodes);
              (*planResult->rightToNode)[joinGroups[jg].second].fill(false);
            }

            // grab the vector for the key tid
            (*planResult->rightToNode)[joinGroups[jg].second][node] = true;
          }

          /// 1.2 Store the join group
          (*planResult->joinGroupsPerNode)[node].push_back(std::make_pair(joinGroups[jg].first, joinGroups[jg].second));
        }
      }
    }

    // set the main record of the page
    getRecord(planResult);

    // store the choice
    selectedAlgorithm = AlgorithmID::AGG_FIRST_ONLY;
  }
}

void pdb::JoinAggPlanner::doFullPlanning(const std::vector<char> &lhsRecordPositions,
                                         const std::vector<char> &rhsRecordPositions,
                                         const std::vector<std::vector<int32_t>> &aggregationGroups,
                                         const std::vector<std::pair<int32_t, int32_t>> &joinGroups) {
  pdb::GreedyPlanner::costs_t c{};
  c.agg_cost = 1;
  c.join_cost = 1;
  c.join_rec_size = 1;
  c.send_coef = 1;
  c.rhs_input_rec_size = 1;
  c.lhs_input_rec_size = 1;
  c.aggregation_rec_size = 1;

  // init the planner run the agg only planner
  pdb::GreedyPlanner planner(numNodes, c, lhsRecordPositions, rhsRecordPositions, aggregationGroups, joinGroups);

  // run for a number of iterations
  planner.run();

  // get the result of the planning
  auto result = planner.get_result();

  // update the planning costs
  planning_costs[(int32_t) AlgorithmID::FULL] = planner.getCost();

  // spin while all the other planners don't finish
  num_finished++;
  while (num_finished != 3) {}

  // the algorithm we selected
  auto choice = selectAlgorithm();

  //
  if (choice == AlgorithmID::JOIN_FIRST_ONLY) {

    // repin the page
    pageToStore->repin();
    UseTemporaryAllocationBlock blk{pageToStore->getBytes(), pageToStore->getSize()};

    // make the plan result object
    Handle<PipJoinAggPlanResult> planResult = pdb::makeObject<PipJoinAggPlanResult>(numNodes);

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

    // go through each join group
    for(auto jg = 0; jg < joinGroups.size(); ++jg) {

      for(auto node = 0; node < numNodes; ++node) {
        if (result.join_groups_to_node[jg * numNodes + node]) {

          /// 1.0 Store the left side
          {
            // make sure we have it
            if ((*planResult->leftToNode).count(joinGroups[jg].first) == 0) {
              (*planResult->leftToNode)[joinGroups[jg].first] = Vector<bool>(numNodes, numNodes);
              (*planResult->leftToNode)[joinGroups[jg].first].fill(false);
            }

            // grab the vector for the key tid
            (*planResult->leftToNode)[joinGroups[jg].first][node] = true;
          }

          /// 1.1 Store the right side
          {
            // make sure we have it
            if ((*planResult->rightToNode).count(joinGroups[jg].second) == 0) {
              (*planResult->rightToNode)[joinGroups[jg].second] = Vector<bool>(numNodes, numNodes);
              (*planResult->rightToNode)[joinGroups[jg].second].fill(false);
            }

            // grab the vector for the key tid
            (*planResult->rightToNode)[joinGroups[jg].second][node] = true;
          }

          /// 1.2 Store the join group
          (*planResult->joinGroupsPerNode)[node].push_back(std::make_pair(joinGroups[jg].first, joinGroups[jg].second));
        }
      }
    }

    // set the main record of the page
    getRecord(planResult);

    // store the choice
    selectedAlgorithm = AlgorithmID::AGG_FIRST_ONLY;
  }
}

pdb::JoinAggPlanner::AlgorithmID pdb::JoinAggPlanner::selectAlgorithm() {

  // we prefer the agg first planning since it does not have a later shuffling phase
  if (planning_costs[(int32_t) AlgorithmID::AGG_FIRST_ONLY] <= planning_costs[(int32_t) AlgorithmID::JOIN_FIRST_ONLY] &&
      planning_costs[(int32_t) AlgorithmID::AGG_FIRST_ONLY] <= planning_costs[(int32_t) AlgorithmID::FULL]) {
    return AlgorithmID::AGG_FIRST_ONLY;
  } else if (planning_costs[(int32_t) AlgorithmID::JOIN_FIRST_ONLY] < planning_costs[(int32_t) AlgorithmID::FULL]) {
    return AlgorithmID::JOIN_FIRST_ONLY;
  } else {
    return AlgorithmID::FULL;
  }
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
}

