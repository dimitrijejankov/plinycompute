//
// Created by dimitrije on 2/22/19.
//

#include <map>

#include <physicalOptimizer/PDBJoinPhysicalNode.h>
#include <physicalOptimizer/PDBAbstractPhysicalNode.h>
#include <physicalAlgorithms/PDBShuffleForJoinAlgorithm.h>
#include <physicalAlgorithms/PDBBroadcastForJoinAlgorithm.h>

PDBPipelineType pdb::PDBJoinPhysicalNode::getType() {
  return PDB_JOIN_SIDE_PIPELINE;
}

pdb::PDBPlanningResult pdb ::PDBJoinPhysicalNode::generateAlgorithm(const std::map<std::string, OptimizerSource> &sourcesWithIDs) {

  // this is the page set we are scanning
  pdb::Handle<PDBSourcePageSetSpec> source = getSourcePageSet();
  pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> additionalSources = makeObject<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>>();

  // generate the algorithm
  return generateAlgorithm(pipeline.front()->getOutputName(), source, sourcesWithIDs, additionalSources);
}

pdb::PDBPlanningResult pdb::PDBJoinPhysicalNode::generatePipelinedAlgorithm(const std::string &startTupleSet,
                                                                            const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                                            const std::map<std::string, OptimizerSource> &sourcesWithIDs,
                                                                            pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources) {
  // generate the algorithm
  return generateAlgorithm(startTupleSet, source, sourcesWithIDs, additionalSources);
}

pdb::PDBPlanningResult pdb::PDBJoinPhysicalNode::generateAlgorithm(const std::string &startTupleSet,
                                                                   const pdb::Handle<pdb::PDBSourcePageSetSpec> &source,
                                                                   const std::map<std::string, pdb::OptimizerSource> &sourcesWithIDs,
                                                                   pdb::Handle<pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>>> &additionalSources) {
  // check if the node is not processed
  assert(state == PDBJoinPhysicalNodeState::PDBJoinPhysicalNodeNotProcessed);

  // just grab the ptr for the other side
  auto otherSidePtr = (PDBJoinPhysicalNode*) otherSide.lock().get();

  // if the other side has been broad casted then this is really cool and we can pipeline through this node
  if(otherSidePtr->state == PDBJoinPhysicalNodeBroadcasted) {

    // create the additional sources
    additionalSources->push_back(otherSidePtr->getSourcePageSet());

    // pipeline this node to the next
    generatePipelinedAlgorithm(startTupleSet, source, sourcesWithIDs, additionalSources);
  }

  // the sink is basically the last computation in the pipeline
  pdb::Handle<PDBSinkPageSetSpec> sink = pdb::makeObject<PDBSinkPageSetSpec>();
  sink->pageSetIdentifier = std::make_pair(computationID, (String) pipeline.back()->getOutputName());

  pdb::Handle<PDBSinkPageSetSpec> intermediate = pdb::makeObject<PDBSinkPageSetSpec>();
  intermediate->pageSetIdentifier = std::make_pair(computationID, (String) (pipeline.back()->getOutputName() + "_to_broadcast"));

  // check if we can broadcast this side (the other side is not shuffled and this side is small enough)
  auto it = sourcesWithIDs.find(getNodeIdentifier());
  if(it->second.first < SHUFFLE_JOIN_THRASHOLD && otherSidePtr->state == PDBJoinPhysicalNodeNotProcessed) {

    // set the type of the sink
    sink->sinkType = PDBSinkType::BroadcastJoinSink;
    intermediate->sinkType = PDBSinkType::BroadcastIntermediateJoinSink;

    // ok so we have to shuffle this side, generate the algorithm
    pdb::Handle<PDBBroadcastForJoinAlgorithm> algorithm = pdb::makeObject<PDBBroadcastForJoinAlgorithm>(startTupleSet,
                                                                                                        pipeline.back()->getOutputName(),
                                                                                                        source,
                                                                                                        intermediate,
                                                                                                        sink,
                                                                                                        additionalSources);

    // mark the state of this node as broadcasted
    state = PDBJoinPhysicalNodeBroadcasted;

    // return the algorithm
    return std::make_pair(algorithm, consumers);
  }

  // set the type of the sink
  sink->sinkType = PDBSinkType::JoinShuffleSink;
  intermediate->sinkType = PDBSinkType::JoinShuffleIntermediateSink;

  // ok so we have to shuffle this side, generate the algorithm
  pdb::Handle<PDBShuffleForJoinAlgorithm> algorithm = pdb::makeObject<PDBShuffleForJoinAlgorithm>(startTupleSet,
                                                                                                  pipeline.back()->getOutputName(),
                                                                                                  source,
                                                                                                  intermediate,
                                                                                                  sink,
                                                                                                  additionalSources);

  // mark the state of this node as shuffled
  state = PDBJoinPhysicalNodeShuffled;

  // figure out if we have new sources
  std::list<PDBAbstractPhysicalNodePtr> newSources;
  if(otherSidePtr->state == PDBJoinPhysicalNodeShuffled) {
    newSources.insert(newSources.begin(), consumers.begin(), consumers.end());
  }

  // return the algorithm and the nodes that consume it's result
  return std::make_pair(algorithm, newSources);
}

// set this value to some reasonable value // TODO this needs to be smarter
const size_t pdb::PDBJoinPhysicalNode::SHUFFLE_JOIN_THRASHOLD = 0;

