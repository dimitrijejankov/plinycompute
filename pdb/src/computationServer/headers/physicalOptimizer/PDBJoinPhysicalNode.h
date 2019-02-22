//
// Created by dimitrije on 2/22/19.
//

#ifndef PDB_PDBJOINPHYSICALNODE_H
#define PDB_PDBJOINPHYSICALNODE_H

#include <PDBAbstractPhysicalNode.h>

namespace pdb {

 class PDBJoinPhysicalNode : public pdb::PDBAbstractPhysicalNode  {

 public:

   PDBJoinPhysicalNode(const std::vector<AtomicComputationPtr>& pipeline, size_t currentNodeIndex) : PDBAbstractPhysicalNode(pipeline, currentNodeIndex) {};

  PDBPipelineType getType() override;

};

}

#endif //PDB_PDBJOINPHYSICALNODE_H
