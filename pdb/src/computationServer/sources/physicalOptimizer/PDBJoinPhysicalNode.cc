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

  // check if the node is not processed
  assert(state == PDBJoinPhysicalNodeState::PDBJoinPhysicalNodeNotProcessed);

  // just grab the ptr for the other side
  auto otherSidePtr = (PDBJoinPhysicalNode*) otherSide.lock().get();

  // if the other side has been broad casted then this is really cool and we can pipeline through this node
  if(otherSidePtr->state == PDBJoinPhysicalNodeBroadcasted) {

    // create the additional sources
    pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> additionalSources = makeObject<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>>();
    additionalSources->push_back(otherSidePtr->getSourcePageSet());

    // pipeline this node to the next
    generatePipelinedAlgorithm(pipeline.front()->getOutputName(), source, additionalSources);
  }

  // the sink is basically the last computation in the pipeline
  pdb::Handle<PDBSinkPageSetSpec> sink = pdb::makeObject<PDBSinkPageSetSpec>();
  sink->sinkType = PDBSinkType::SetSink;
  sink->pageSetIdentifier = std::make_pair(computationID, (String) pipeline.back()->getOutputName());

  // check if we can broadcast this side (the other side is not shuffled and this side is small enough)
  auto it = sourcesWithIDs.find(getNodeIdentifier());
  if(it->second.first < SHUFFLE_JOIN_THRASHOLD && otherSidePtr->state == PDBJoinPhysicalNodeShuffled) {

    // ok so we have to shuffle this side, generate the algorithm
    pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> additionalSources = nullptr;
    pdb::Handle<PDBBroadcastForJoinAlgorithm> algorithm = pdb::makeObject<PDBBroadcastForJoinAlgorithm>(pipeline.front()->getOutputName(),
                                                                                                        pipeline.back()->getOutputName(),
                                                                                                        source,
                                                                                                        sink,
                                                                                                        additionalSources);

    // mark the state of this node as broadcasted
    state = PDBJoinPhysicalNodeBroadcasted;

    // return the algorithm
    return std::make_pair(algorithm, consumers);
  }

  // ok so we have to shuffle this side, generate the algorithm
  pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> additionalSources = nullptr;
  pdb::Handle<PDBShuffleForJoinAlgorithm> algorithm = pdb::makeObject<PDBShuffleForJoinAlgorithm>(pipeline.front()->getOutputName(),
                                                                                                  pipeline.back()->getOutputName(),
                                                                                                  source,
                                                                                                  sink,
                                                                                                  additionalSources);

  // mark the state of this node as shuffled
  state = PDBJoinPhysicalNodeShuffled;

  // return the algorithm and the nodes that consume it's result
  return std::make_pair(algorithm, consumers);
}

pdb::PDBPlanningResult pdb::PDBJoinPhysicalNode::generatePipelinedAlgorithm(const std::string &startTupleSet,
                                                                            const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                                            pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources) {
  return pdb::PDBPlanningResult();
}



const size_t pdb::PDBJoinPhysicalNode::SHUFFLE_JOIN_THRASHOLD = 0;