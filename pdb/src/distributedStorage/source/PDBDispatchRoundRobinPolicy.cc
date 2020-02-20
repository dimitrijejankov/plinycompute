#include "PDBDispatchRoundRobinPolicy.h"

namespace pdb {

PDBCatalogNodePtr PDBDispatchRoundRobinPolicy::getNextNode(const std::string &database,
                                                           const std::string &set,
                                                           const std::vector<PDBCatalogNodePtr> &nodes) {

  // get the next index
  auto nodeIndex = nextRoundRobin[std::make_pair(database, set)];

  // set the node
  nextRoundRobin[std::make_pair(database, set)] = nodeIndex + 1;

  // return the node
  return nodes[nodeIndex % nodes.size()];
}

}