//
// Created by dimitrije on 2/22/19.
//

#ifndef PDB_PDBJOINPHYSICALNODE_H
#define PDB_PDBJOINPHYSICALNODE_H

#include <PDBAbstractPhysicalNode.h>

namespace pdb {

class PDBJoinPhysicalNode : public pdb::PDBAbstractPhysicalNode {

 public:

  PDBJoinPhysicalNode(const std::vector<AtomicComputationPtr> &pipeline, size_t computationID, size_t currentNodeIndex)
      : PDBAbstractPhysicalNode(pipeline, computationID, currentNodeIndex) {};

  PDBPipelineType getType() override;

  PDBPlanningResult generateAlgorithm() override;

  PDBPlanningResult generatePipelinedAlgorithm(const std::string &startTupleSet,
                                                 const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                 pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> &additionalSources) override;

};

}

#endif //PDB_PDBJOINPHYSICALNODE_H
