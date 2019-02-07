//
// Created by dimitrije on 2/7/19.
//

#include <PDBDispatcherRandomPolicy.h>

namespace pdb {

PDBCatalogNodePtr PDBDispatcherRandomPolicy::getNextNode(const std::string &database, const std::string &set, const std::vector<PDBCatalogNodePtr> &nodes) {

  // lock the thing
  std::unique_lock<std::mutex> lck(m);

  // figure out the which node
  auto node = generator() % nodes.size();

  // return the node
  return nodes[node];
}

}

