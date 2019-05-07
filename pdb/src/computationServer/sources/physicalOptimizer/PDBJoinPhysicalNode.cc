//
// Created by dimitrije on 2/22/19.
//

#include <physicalOptimizer/PDBJoinPhysicalNode.h>
#include <physicalOptimizer/PDBAbstractPhysicalNode.h>

#include "physicalOptimizer/PDBJoinPhysicalNode.h"


PDBPipelineType pdb::PDBJoinPhysicalNode::getType() {
  return PDB_JOIN_SIDE_PIPELINE;
}

pdb::PDBPlanningResult pdb ::PDBJoinPhysicalNode::generateAlgorithm() {
  return pdb::PDBPlanningResult();
}

pdb::PDBPlanningResult pdb::PDBJoinPhysicalNode::generatePipelinedAlgorithm(const std::string &startTupleSet,
                                                                            const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                                            pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> &additionalSources) {
  return pdb::PDBPlanningResult();
}
