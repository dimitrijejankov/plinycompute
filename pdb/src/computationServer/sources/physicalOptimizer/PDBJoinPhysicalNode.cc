//
// Created by dimitrije on 2/22/19.
//

#include <physicalOptimizer/PDBJoinPhysicalNode.h>
#include <physicalOptimizer/PDBAbstractPhysicalNode.h>

#include "physicalOptimizer/PDBJoinPhysicalNode.h"


PDBPipelineType pdb::PDBJoinPhysicalNode::getType() {
  return PDB_JOIN_SIDE_PIPELINE;
}

pdb::Handle<pdb::PDBPhysicalAlgorithm> pdb::PDBJoinPhysicalNode::generateAlgorithm() {
  return nullptr;
}
