//
// Created by dimitrije on 2/21/19.
//

#include <physicalAlgorithms/PDBAggregationPipeAlgorithm.h>
#include "physicalOptimizer/PDBAggregationPhysicalNode.h"


namespace pdb {

PDBPipelineType pdb::PDBAggregationPhysicalNode::getType() {
  return PDB_AGGREGATION_PIPELINE;
}

pdb::PDBPlanningResult PDBAggregationPhysicalNode::generateAlgorithm(const std::string &firstTupleSet,
                                                                     const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                                     sourceCosts &sourcesWithIDs,
                                                                     pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources,
                                                                     bool shouldSwapLeftAndRight) {

  // the aggregation has two parts, one part packs the records into a bunch of hash tables
  // the second part does the actual aggregation, both parts are run at the same time
  // the aggregation starts by scanning a source tuple set, packs the records into a bunch of hash tables and then sends
  // them to the appropriate node, then the second part of the pipeline aggregate stuff

  // this is the page set that is containing the bunch of hash maps want to send
  pdb::Handle<PDBSinkPageSetSpec> hashedToSend = pdb::makeObject<PDBSinkPageSetSpec>();
  hashedToSend->sinkType = PDBSinkType::AggShuffleSink;
  hashedToSend->pageSetIdentifier = std::make_pair(computationID, (String) (pipeline.back()->getOutputName()  + "_hashed_to_send"));

  // this is the page set where we put the hash maps send over the wire
  pdb::Handle<PDBSourcePageSetSpec> hashedToRecv = pdb::makeObject<PDBSourcePageSetSpec>();
  hashedToRecv->sourceType = PDBSourceType::ShuffledAggregatesSource;
  hashedToRecv->pageSetIdentifier = std::make_pair(computationID, (String) (pipeline.back()->getOutputName() + "_hashed_to_recv"));

  // this is the tuple set where we put the output
  pdb::Handle<PDBSinkPageSetSpec> sink = pdb::makeObject<PDBSinkPageSetSpec>();
  sink->sinkType = PDBSinkType::AggregationSink;
  sink->pageSetIdentifier = std::make_pair(computationID, (String) pipeline.back()->getOutputName());

  // just store the sink page set for later use by the eventual consumers
  setSinkPageSet(sink);

  // create the algorithm
  pdb::Handle<PDBAggregationPipeAlgorithm> algorithm = pdb::makeObject<PDBAggregationPipeAlgorithm>(firstTupleSet,
                                                                                                    pipeline.back()->getOutputName(),
                                                                                                    source,
                                                                                                    hashedToSend,
                                                                                                    hashedToRecv,
                                                                                                    sink,
                                                                                                    additionalSources,
                                                                                                    shouldSwapLeftAndRight);
  // return the stuff
  return std::make_pair(algorithm, consumers);
}

pdb::PDBPlanningResult PDBAggregationPhysicalNode::generatePipelinedAlgorithm(const std::string &firstTupleSet,
                                                                              const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                                              sourceCosts &sourcesWithIDs,
                                                                              pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources,
                                                                              bool shouldSwapLeftAndRight) {

  // this is the same as @see generateAlgorithm except now the source is the source of the pipe we pipelined to this
  // and the additional source are transferred for that pipeline. We can not pipeline an aggregation

  return generateAlgorithm(firstTupleSet, source, sourcesWithIDs, additionalSources, shouldSwapLeftAndRight);
}

}