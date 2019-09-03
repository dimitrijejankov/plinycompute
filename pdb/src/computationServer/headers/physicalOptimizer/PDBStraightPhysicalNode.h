//
// Created by dimitrije on 2/21/19.
//

#ifndef PDB_PDBSTRAIGHTPIPELINE_H
#define PDB_PDBSTRAIGHTPIPELINE_H

#include "PDBAbstractPhysicalNode.h"

namespace pdb {

class PDBStraightPhysicalNode : public PDBAbstractPhysicalNode {
public:

  PDBStraightPhysicalNode(const std::vector<AtomicComputationPtr>& pipeline, size_t computationID, size_t currentNodeIndex) : PDBAbstractPhysicalNode(pipeline, computationID, currentNodeIndex) {};

  PDBPipelineType getType() override;

  pdb::PDBPlanningResult generateAlgorithm(PDBAbstractPhysicalNodePtr &child,
                                           PDBPageSetCosts &pageSetCosts) override;
};

}


#endif //PDB_PDBSTRAIGHTPIPELINE_H
