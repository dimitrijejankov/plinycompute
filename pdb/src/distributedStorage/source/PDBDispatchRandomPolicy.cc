//
// Created by dimitrije on 2/7/19.
//

#include <PDBDispatchRandomPolicy.h>

namespace pdb {

PDBCatalogNodePtr PDBDispatchRandomPolicy::getNextNode(const std::string &database, const std::string &set, const std::vector<PDBCatalogNodePtr> &nodes) {

  // lock the thing
  std::unique_lock<std::mutex> lck(m);

  // figure out the which node
  auto node = generator() % nodes.size();

  // return the node
  return nodes[node];
}

}

