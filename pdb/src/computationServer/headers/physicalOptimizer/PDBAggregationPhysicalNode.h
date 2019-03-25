//
// Created by dimitrije on 2/21/19.
//

#ifndef PDB_PDBAGGREGATIONPIPELINE_H
#define PDB_PDBAGGREGATIONPIPELINE_H

#include "PDBAbstractPhysicalNode.h"

namespace pdb {

class PDBAggregationPhysicalNode : public PDBAbstractPhysicalNode  {

public:
  
  PDBAggregationPhysicalNode(const std::vector<AtomicComputationPtr>& pipeline, size_t computationID, size_t currentNodeIndex) : PDBAbstractPhysicalNode(pipeline, computationID, currentNodeIndex) {};

  ~PDBAggregationPhysicalNode() override = default;

  PDBPipelineType getType() override;

  PDBPlanningResult generateAlgorithm() override;

  PDBPlanningResult generatePipelinedAlgorithm(const std::string &firstTupleSet,
                                                 const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                 pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> &additionalSources) override;

};

}


#endif //PDB_PDBAGGREGATIONPIPELINE_H
