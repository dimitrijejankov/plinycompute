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

  Handle<PDBPhysicalAlgorithm> generateAlgorithm() override;

};

}

#endif //PDB_PDBJOINPHYSICALNODE_H
