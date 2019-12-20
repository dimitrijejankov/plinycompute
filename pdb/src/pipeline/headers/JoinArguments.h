#pragma once

#include <utility>
#include <PDBAbstractPageSet.h>
#include <ComputeInfo.h>
#include <JoinAggSideSender.h>
#include <cassert>

namespace pdb {

// the shuffle join arguments
class ShuffleJoinArg : public pdb::ComputeInfo {
public:

  // the constructor
  explicit ShuffleJoinArg(bool swapLeftAndRightSide) : swapLeftAndRightSide(swapLeftAndRightSide) {}

  // should we swap the left and the right side in the tcap
  bool swapLeftAndRightSide;
};

// the join agg arguments
class JoinAggSideArg : public pdb::ComputeInfo {
 public:

  // the constructor
  explicit JoinAggSideArg(pdb::PDBPageHandle keyToNode,
                          std::shared_ptr<std::vector<JoinAggSideSenderPtr>> &senders,
                          pdb::PDBPageHandle plan,
                          bool isLeft) : keyToNode(std::move(keyToNode)),
                                         plan(std::move(plan)),
                                         senders(senders),
                                         isLeft(isLeft) {}

  // key to node map is on this page
  pdb::PDBPageHandle keyToNode;

  // the senders to nodes
  std::shared_ptr<std::vector<JoinAggSideSenderPtr>> senders;

  // the plan page
  pdb::PDBPageHandle plan;

  // is this the left or right side
  bool isLeft;
};
using JoinAggSideArgPtr = std::shared_ptr<JoinAggSideArg>;

// used to parameterize joins that are run as part of a pipeline
class JoinArg {
public:

  // init the join arguments
  explicit JoinArg(PDBAbstractPageSetPtr hashTablePageSet) : hashTablePageSet(std::move(hashTablePageSet)) {}

  // the location of the hash table
  PDBAbstractPageSetPtr hashTablePageSet;

};
using JoinArgPtr = std::shared_ptr<JoinArg>;

// basically we bundle all join arguments together
class JoinArguments : public pdb::ComputeInfo {
public:

  JoinArguments() = default;
  JoinArguments(std::initializer_list<std::pair<const std::string, JoinArgPtr>> l) : hashTables(l) {}
  explicit JoinArguments(unordered_map<string, JoinArgPtr> hashTables) : hashTables(std::move(hashTables)) {}

  // the list of hash tables
  std::unordered_map<std::string, JoinArgPtr> hashTables;

  // is join agg side (used for the join agg algorithm)
  bool isJoinAggSide = false;
};

using JoinArgumentsPtr = std::shared_ptr<JoinArguments>;
using JoinArgumentsInit = std::initializer_list<std::pair<const std::string, JoinArgPtr>>;

class KeyJoinSourceArgs : public pdb::ComputeInfo {
public:

  explicit KeyJoinSourceArgs(const std::vector<PDBPageHandle> &init) {

    // make sure we have two
    assert(init.size() == 2);

    // init the page set
    lhsTablePageSet = init[0];
    rhsTablePageSet = init[1];
  }

  // the location of the lhs key table
  PDBPageHandle lhsTablePageSet;

  // the location of the rhs key table
  PDBPageHandle rhsTablePageSet;
};

using KeyJoinSourceArgsPtr = std::shared_ptr<KeyJoinSourceArgs>;

}