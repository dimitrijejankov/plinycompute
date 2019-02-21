//
// Created by dimitrije on 2/21/19.
//

#include "PDBStraightPhysicalNode.h"

PDBPipelineType pdb::PDBStraightPhysicalNode::getType() {
  return PDB_STRAIGHT_PIPELINE;
}

std::string pdb::PDBStraightPhysicalNode::getNodeIdentifier() {
  return "";
}
