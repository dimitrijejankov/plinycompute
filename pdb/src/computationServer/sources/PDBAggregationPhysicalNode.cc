//
// Created by dimitrije on 2/21/19.
//

#include "../headers/PDBAggregationPhysicalNode.h"

PDBPipelineType pdb::PDBAggregationPhysicalNode::getType() {
  return PDB_AGGREGATION_PIPELINE;
}

std::string pdb::PDBAggregationPhysicalNode::getNodeIdentifier() {
  return "";
}
