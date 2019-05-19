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

  pdb::PDBPlanningResult generatePipelinedAlgorithm(const std::string &startTupleSet,
                                                    const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                    PDBPageSetCosts &sourcesWithIDs,
                                                    pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources,
                                                    bool shouldSwapLeftAndRight) override;

  pdb::PDBPlanningResult generateAlgorithm(const std::string &startTupleSet,
                                             const pdb::Handle<PDBSourcePageSetSpec> &source,
                                             PDBPageSetCosts &sourcesWithIDs,
                                             pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources,
                                             bool shouldSwapLeftAndRight) override;
};

}


#endif //PDB_PDBSTRAIGHTPIPELINE_H
