//
// Created by dimitrije on 2/22/19.
//

#ifndef PDB_PDBJOINPHYSICALNODE_H
#define PDB_PDBJOINPHYSICALNODE_H

#include "PDBOptimizerSource.h"
#include <PDBAbstractPhysicalNode.h>
#include <map>
#include <string>

namespace pdb {

enum PDBJoinPhysicalNodeState {

  PDBJoinPhysicalNodeNotProcessed,
  PDBJoinPhysicalNodeBroadcasted,
  PDBJoinPhysicalNodeShuffled
};

class PDBJoinPhysicalNode : public pdb::PDBAbstractPhysicalNode {

public:

  PDBJoinPhysicalNode(const std::vector<AtomicComputationPtr> &pipeline, size_t computationID, size_t currentNodeIndex)
      : PDBAbstractPhysicalNode(pipeline, computationID, currentNodeIndex) {};

  PDBPipelineType getType() override;

  pdb::PDBPlanningResult generateAlgorithm(const AtomicComputationPtr &startAtomicComputation,
                                           const pdb::Handle<PDBSourcePageSetSpec> &source,
                                           PDBPageSetCosts &sourcesWithIDs,
                                           pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources,
                                           bool shouldSwapLeftAndRight) override;

  pdb::PDBPlanningResult generatePipelinedAlgorithm(const AtomicComputationPtr &startAtomicComputation,
                                                    const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                    PDBPageSetCosts &sourcesWithIDs,
                                                    pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources,
                                                    bool shouldSwapLeftAndRight) override;


  /**
   * The other side
   */
  pdb::PDBAbstractPhysicalNodeWeakPtr otherSide;

private:

  /**
   * This constant is the cutoff threshold point where we use the shuffle join instead of the broadcast join
   */
  static const size_t SHUFFLE_JOIN_THRASHOLD;

  /**
   * The state of the node
   */
  PDBJoinPhysicalNodeState state = PDBJoinPhysicalNodeNotProcessed;
};

}

#endif //PDB_PDBJOINPHYSICALNODE_H
