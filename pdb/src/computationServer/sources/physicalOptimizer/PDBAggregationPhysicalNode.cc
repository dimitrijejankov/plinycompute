//
// Created by dimitrije on 2/21/19.
//

#include <physicalAlgorithms/PDBAggregationPipeAlgorithm.h>
#include "physicalOptimizer/PDBAggregationPhysicalNode.h"

PDBPipelineType pdb::PDBAggregationPhysicalNode::getType() {
  return PDB_AGGREGATION_PIPELINE;
}

pdb::Handle<pdb::PDBPhysicalAlgorithm> pdb::PDBAggregationPhysicalNode::generateAlgorithm() {

  pdb::Handle<PDBSourcePageSetSpec> source = pdb::makeObject<PDBSourcePageSetSpec>();
  source->tupleSetIdentifier = pipeline.front()->getOutputName();
  source->sourceType = PDBSourceType::SetScanSource;
  source->pageSetIdentifier = std::make_pair(computationID, (String) pipeline.front()->getOutputName());

  pdb::Handle<PDBSinkPageSetSpec> sink = pdb::makeObject<PDBSinkPageSetSpec>();
  sink->tupleSetIdentifier = pipeline.back()->getOutputName();
  sink->sinkType = PDBSinkType::SetSink;
  sink->pageSetIdentifier = std::make_pair(computationID, (String) pipeline.back()->getOutputName());

  pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> vector = nullptr;
  pdb::Handle<PDBAggregationPipeAlgorithm> algorithm = pdb::makeObject<PDBAggregationPipeAlgorithm>(source, sink, vector);

  return algorithm;
}
