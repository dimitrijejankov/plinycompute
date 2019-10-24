//
// Created by dimitrije on 2/21/19.
//

#ifndef PDB_PDBAGGREGATIONPIPELINE_H
#define PDB_PDBAGGREGATIONPIPELINE_H

#include "PDBAbstractPhysicalNode.h"

namespace pdb {

class PDBAggregationPhysicalNode : public PDBAbstractPhysicalNode  {

public:
  
  PDBAggregationPhysicalNode(const std::vector<AtomicComputationPtr>& pipeline,
                             size_t computationID,
                             size_t currentNodeIndex,
                             bool keyed) : PDBAbstractPhysicalNode(pipeline,
                                                                   computationID,
                                                                   currentNodeIndex,
                                                                   keyed) {};

  ~PDBAggregationPhysicalNode() override = default;

  PDBPipelineType getType() override;

  pdb::PDBPlanningResult generateAlgorithm(PDBAbstractPhysicalNodePtr &child,
                                           PDBPageSetCosts &pageSetCosts) override;

  pdb::PDBPlanningResult generateMergedAlgorithm(const PDBAbstractPhysicalNodePtr &lhs,
                                                 const PDBAbstractPhysicalNodePtr &rhs,
                                                 const PDBPageSetCosts &pageSetCosts);

};

}


#endif //PDB_PDBAGGREGATIONPIPELINE_H
