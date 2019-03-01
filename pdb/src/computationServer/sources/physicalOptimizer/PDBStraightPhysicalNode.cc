//
// Created by dimitrije on 2/21/19.
//

#include "physicalOptimizer/PDBStraightPhysicalNode.h"

PDBPipelineType pdb::PDBStraightPhysicalNode::getType() {
  return PDB_STRAIGHT_PIPELINE;
}

pdb::Handle<pdb::PDBPhysicalAlgorithm> pdb::PDBStraightPhysicalNode::generateAlgorithm() {

  return nullptr;
}
