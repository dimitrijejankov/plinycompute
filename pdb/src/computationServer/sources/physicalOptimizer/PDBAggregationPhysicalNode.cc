//
// Created by dimitrije on 2/21/19.
//

#include "physicalOptimizer/PDBAggregationPhysicalNode.h"

PDBPipelineType pdb::PDBAggregationPhysicalNode::getType() {
  return PDB_AGGREGATION_PIPELINE;
}

pdb::Handle<pdb::PDBPhysicalAlgorithm> pdb::PDBAggregationPhysicalNode::generateAlgorithm() {
  return nullptr;
}
