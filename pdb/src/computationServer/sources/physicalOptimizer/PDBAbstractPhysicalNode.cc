//
// Created by dimitrije on 2/21/19.
//

#include <PDBAbstractPhysicalNode.h>


const std::list<pdb::PDBAbstractPhysicalNodePtr> &pdb::PDBAbstractPhysicalNode::getConsumers() {
  return consumers;
}

const std::list<pdb::PDBAbstractPhysicalNodePtr> pdb::PDBAbstractPhysicalNode::getProducers() {

  // create the list
  std::list<PDBAbstractPhysicalNodePtr> out;

  // fill up the list
  for(auto &it : producers) {
    out.push_back(it.lock());
  }

  // return the list
  return std::move(out);
}