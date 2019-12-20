#pragma once

#include "PDBAnonymousPageSet.h"
#include "PDBVector.h"
#include "PipJoinAggPlanResult.h"

namespace pdb {

class JoinAggPlanner {

  // the page containing the page
  pdb::PDBPageHandle joinAggPageSet;

  // the first integer is the join key value identifier, the second value is the node it comes from
  using TIDType = std::pair<uint32_t, int32_t>;

  // this is a vector of keys that are joined to form an aggregation group
  using TIDVector = Vector<std::pair<TIDType, TIDType>>;

  // each aggregation group is identified by an unsigned integer
  // this maps maps the aggregation grup to all the keys that are joined to form it
  using TIDIndexMap = Map<uint32_t, TIDVector>;

  // the input page
  PDBPageHandle inputPage;

  // the the map
  Handle<TIDIndexMap> joinGroups;

  // the number of nodes
  int32_t numNodes;

  //  the number of threads for each node
  int32_t numThreads;

  // the page where we store the result
  PDBPageHandle pageToStore;

public:

  explicit JoinAggPlanner(const pdb::PDBAnonymousPageSetPtr &joinAggPageSet,
                          uint32_t numNodes,
                          uint32_t numThreads,
                          const PDBPageHandle& pageToStore);

  void doPlanning();

  void print(const Handle<PipJoinAggPlanResult> &planResult);
};

}