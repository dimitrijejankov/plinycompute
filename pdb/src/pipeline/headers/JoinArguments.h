//
// Created by dimitrije on 3/30/19.
//

#ifndef PDB_JOINARGUMENTS_H
#define PDB_JOINARGUMENTS_H

namespace pdb {

// used to parameterize joins that are run as part of a pipeline
class JoinArg {
 public:
  ComputePlan& plan;
  // the location of the hash table // TODO this needs to be a page set or something
  void *pageWhereHashTableIs;
};
using JoinArgPtr = std::shared_ptr<JoinArg>;

// basically we bundle all join arguments together
class JoinArguments : ComputeInfo {
public:
  // the list of hash tables
  std::unordered_map<std::string, JoinArgPtr> hashTables;
};

using JoinArgumentsPtr = std::shared_ptr<JoinArguments>;

}

#endif //PDB_JOINARGUMENTS_H
