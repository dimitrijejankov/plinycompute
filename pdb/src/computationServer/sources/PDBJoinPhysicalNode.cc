//
// Created by dimitrije on 2/22/19.
//

#include <PDBJoinPhysicalNode.h>
#include <PDBAbstractPhysicalNode.h>

#include "PDBJoinPhysicalNode.h"


PDBPipelineType pdb::PDBJoinPhysicalNode::getType() {
  return PDB_JOIN_SIDE_PIPELINE;
}
