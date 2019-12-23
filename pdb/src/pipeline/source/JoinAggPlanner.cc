#include "JoinAggPlanner.h"

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

  // the current node
  int32_t curNod = 0;

  // go through the map and do two things
  // assign aggregation groups to nodes
  for(auto it = this->aggGroups->begin(); it != this->aggGroups->end(); ++it) {

    /// 0. Round robing the aggregation groups

    // assign the aggregation group to the node
    (*planResult->aggToNode)[(*it).key] = curNod;

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
        (*planResult->leftToNode)[joinedTIDs[i].first.first][curNod] = true;
      }

      /// 1.1 Store the right side
      {
        // make sure we have it
        if((*planResult->rightToNode).count(joinedTIDs[i].second.first) == 0) {
          (*planResult->rightToNode)[joinedTIDs[i].second.first] = Vector<bool>(numNodes, numNodes);
          (*planResult->rightToNode)[joinedTIDs[i].second.first].fill(false);
        }

        // grab the vector for the key tid
        (*planResult->rightToNode)[joinedTIDs[i].second.first][curNod] = true;
      }

      /// 1.2 Store the join group
      {
        (*planResult->joinGroupsPerNode)[curNod].push_back(std::make_pair(joinedTIDs[i].first.first, joinedTIDs[i].second.first));
      }
    }

    // round robin the nodes
    curNod = (curNod + 1) % numNodes;
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

