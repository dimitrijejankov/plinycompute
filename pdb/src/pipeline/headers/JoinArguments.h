#include <utility>

#include <utility>

//
// Created by dimitrije on 3/30/19.
//

#ifndef PDB_JOINARGUMENTS_H
#define PDB_JOINARGUMENTS_H

namespace pdb {

// used to parameterize joins that are run as part of a pipeline
class JoinArg {
public:

  explicit JoinArg(PDBAbstractPageSetPtr hashTablePageSet) : hashTablePageSet(std::move(hashTablePageSet)) {}

  // the page set that contains the pages
  PDBAbstractPageSetPtr hashTablePageSet;
};
using JoinArgPtr = std::shared_ptr<JoinArg>;

// basically we bundle all join arguments together
class JoinArguments : public ComputeInfo {
public:

  explicit JoinArguments(unordered_map<string, JoinArgPtr> hashTables) : hashTables(std::move(hashTables)) {}

  // the list of hash tables
  std::unordered_map<std::string, JoinArgPtr> hashTables;
};

using JoinArgumentsPtr = std::shared_ptr<JoinArguments>;

}

#endif //PDB_JOINARGUMENTS_H
