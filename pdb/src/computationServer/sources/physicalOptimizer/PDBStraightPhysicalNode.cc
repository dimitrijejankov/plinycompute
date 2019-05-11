//
// Created by dimitrije on 2/21/19.
//

#include <physicalAlgorithms/PDBStraightPipeAlgorithm.h>
#include <physicalOptimizer/PDBStraightPhysicalNode.h>
#include <PDBVector.h>

PDBPipelineType pdb::PDBStraightPhysicalNode::getType() {
  return PDB_STRAIGHT_PIPELINE;
}

pdb::PDBPlanningResult pdb::PDBStraightPhysicalNode::generatePipelinedAlgorithm(const std::string &startTupleSet,
                                                                                const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                                                sourceCosts &sourcesWithIDs,
                                                                                pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources) {

  // this is the same as @see generateAlgorithm except now the source is the source of the pipe we pipelined to this
  // and the additional source are transferred for that pipeline.
  return generateAlgorithm(startTupleSet, source, sourcesWithIDs, additionalSources);
}

pdb::PDBPlanningResult pdb::PDBStraightPhysicalNode::generateAlgorithm(const std::string &startTupleSet,
                                                                       const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                                       sourceCosts &sourcesWithIDs,
                                                                       pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources) {


  // can we pipeline this guy? we can do that if we only have one consumer
  if(consumers.size() == 1) {
    return consumers.front()->generatePipelinedAlgorithm(startTupleSet, source, sourcesWithIDs, additionalSources);
  }

  // the sink is basically the last computation in the pipeline
  pdb::Handle<PDBSinkPageSetSpec> sink = pdb::makeObject<PDBSinkPageSetSpec>();
  sink->sinkType = PDBSinkType::SetSink;
  sink->pageSetIdentifier = std::make_pair(computationID, (String) pipeline.back()->getOutputName());

  // just store the sink page set for later use by the eventual consumers
  setSinkPageSet(sink);

  // generate the algorithm
  pdb::Handle<PDBStraightPipeAlgorithm> algorithm = pdb::makeObject<PDBStraightPipeAlgorithm>(startTupleSet,
                                                                                              pipeline.back()->getOutputName(),
                                                                                              source,
                                                                                              sink,
                                                                                              additionalSources);

  // return the algorithm and the nodes that consume it's result
  return std::make_pair(algorithm, consumers);
}