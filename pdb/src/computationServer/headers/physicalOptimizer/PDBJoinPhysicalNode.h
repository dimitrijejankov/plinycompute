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

  pdb::PDBPlanningResult generateAlgorithm(PDBAbstractPhysicalNodePtr &child,
                                           PDBPageSetCosts &pageSetCosts) override;


  size_t getPrimarySourcesSize(PDBPageSetCosts &pageSetCosts);

  /**
   * The other side
   */
  pdb::PDBAbstractPhysicalNodeWeakPtr otherSide;

private:

  /**
   * This constant is the cutoff threshold point where we use the shuffle join instead of the broadcast join
   */
  static size_t SHUFFLE_JOIN_THRASHOLD;

  /**
   * The state of the node
   */
  PDBJoinPhysicalNodeState state = PDBJoinPhysicalNodeNotProcessed;

  FRIEND_TEST(TestPhysicalOptimizer, TestJoin1);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin2);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin3);
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregationAfterTwoWayJoin);
};

}

#endif //PDB_PDBJOINPHYSICALNODE_H
