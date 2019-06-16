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

pdb::PDBPlanningResult pdb::PDBJoinPhysicalNode::generatePipelinedAlgorithm(const std::string &startTupleSet,
                                                                            const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                                            sourceCosts &sourcesWithIDs,
                                                                            pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources,
                                                                            bool shouldSwapLeftAndRight) {
  // generate the algorithm
  return generateAlgorithm(startTupleSet, source, sourcesWithIDs, additionalSources, shouldSwapLeftAndRight);
}

pdb::PDBPlanningResult pdb::PDBJoinPhysicalNode::generateAlgorithm(const std::string &startTupleSet,
                                                                   const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                                   sourceCosts &sourcesWithIDs,
                                                                   pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources,
                                                                   bool shouldSwapLeftAndRight) {
  // check if the node is not processed
  assert(state == PDBJoinPhysicalNodeState::PDBJoinPhysicalNodeNotProcessed);

  // just grab the ptr for the other side
  auto otherSidePtr = (PDBJoinPhysicalNode*) otherSide.lock().get();

  // if the other side has been broad casted then this is really cool and we can pipeline through this node
  if(otherSidePtr->state == PDBJoinPhysicalNodeBroadcasted) {

    // make the additional source from the other side
    pdb::Handle<PDBSourcePageSetSpec> additionalSource = pdb::makeObject<PDBSourcePageSetSpec>();
    additionalSource->sourceType = PDBSourceType::BroadcastJoinSource;
    additionalSource->pageSetIdentifier = std::make_pair(computationID, (String) otherSidePtr->pipeline.back()->getOutputName());

    // create the additional sources
    additionalSources->push_back(additionalSource);

    // make sure everything is
    assert(consumers.size() == 1);

    // pipeline this node to the next, it always has to exist and it always has to be one
    return consumers.front()->generatePipelinedAlgorithm(startTupleSet, source, sourcesWithIDs, additionalSources, shouldSwapLeftAndRight);
  }

  // the sink is basically the last computation in the pipeline
  pdb::Handle<PDBSinkPageSetSpec> sink = pdb::makeObject<PDBSinkPageSetSpec>();
  sink->pageSetIdentifier = std::make_pair(computationID, (String) pipeline.back()->getOutputName());

  // check if we can broadcast this side (the other side is not shuffled and this side is small enough)
  auto it = sourcesWithIDs.find(source->pageSetIdentifier);
  if(it->second.first < SHUFFLE_JOIN_THRASHOLD && otherSidePtr->state == PDBJoinPhysicalNodeNotProcessed) {

    // set the type of the sink
    sink->sinkType = PDBSinkType::BroadcastJoinSink;

    // this is the page set that is containing the bunch of hash maps want to send
    pdb::Handle<PDBSinkPageSetSpec> hashedToSend = pdb::makeObject<PDBSinkPageSetSpec>();
    hashedToSend->sinkType = PDBSinkType::BroadcastIntermediateJoinSink;
    hashedToSend->pageSetIdentifier = std::make_pair(computationID, (String)(pipeline.back()->getOutputName() + "_hashed_to_send"));

    pdb::Handle<PDBSourcePageSetSpec> hashedToRecv = pdb::makeObject<PDBSourcePageSetSpec>();
    hashedToRecv->sourceType = PDBSourceType::BroadcastIntermediateJoinSource;
    hashedToRecv->pageSetIdentifier = std::make_pair(computationID, (String)(pipeline.back()->getOutputName() + "_hashed_to_recv"));

    // set this nodes sink specifier
    sinkPageSet.produced = true;
    sinkPageSet.sinkType = BroadcastJoinSink;
    sinkPageSet.pageSetIdentifier = sink->pageSetIdentifier;

    // ok so we have to shuffle this side, generate the algorithm
    pdb::Handle<PDBBroadcastForJoinAlgorithm> algorithm = pdb::makeObject<PDBBroadcastForJoinAlgorithm>(startTupleSet,
                                                                                                        pipeline.back()->getOutputName(),
                                                                                                        source,
                                                                                                        hashedToSend,
                                                                                                        hashedToRecv,
                                                                                                        sink,
                                                                                                        additionalSources,
                                                                                                        shouldSwapLeftAndRight);

    // mark the state of this node as broadcasted
    state = PDBJoinPhysicalNodeBroadcasted;

    // add all the consumed page sets
    std::list<PDBPageSetIdentifier> consumedPageSets = { source->pageSetIdentifier, hashedToSend->pageSetIdentifier, hashedToRecv->pageSetIdentifier };
    for(int i = 0; i < additionalSources->size(); ++i) {
      consumedPageSets.insert(consumedPageSets.begin(), (*additionalSources)[i]->pageSetIdentifier);
    }

    // set the page sets created, the produced page set has to have a page set
    std::vector<std::pair<PDBPageSetIdentifier, size_t>> newPageSets = { std::make_pair(sink->pageSetIdentifier, 1),
                                                                         std::make_pair(hashedToSend->pageSetIdentifier, 1),
                                                                         std::make_pair(hashedToRecv->pageSetIdentifier, 1)};

    // return the algorithm and the nodes that consume it's result
    return std::move(PDBPlanningResult(algorithm, std::list<pdb::PDBAbstractPhysicalNodePtr>(), consumedPageSets, newPageSets));
  }

  // set the type of the sink
  sink->sinkType = PDBSinkType::JoinShuffleSink;

  // create the intermediate page set
  pdb::Handle<PDBSinkPageSetSpec> intermediate = pdb::makeObject<PDBSinkPageSetSpec>();
  intermediate->sinkType = PDBSinkType::JoinShuffleIntermediateSink;
  intermediate->pageSetIdentifier = std::make_pair(computationID, (String) (pipeline.back()->getOutputName() + "_to_shuffle"));

  // set this nodes sink specifier
  sinkPageSet.produced = true;
  sinkPageSet.sinkType = JoinShuffleSink;
  sinkPageSet.pageSetIdentifier = sink->pageSetIdentifier;

  // ok so we have to shuffle this side, generate the algorithm
  pdb::Handle<PDBShuffleForJoinAlgorithm> algorithm = pdb::makeObject<PDBShuffleForJoinAlgorithm>(startTupleSet,
                                                                                                  pipeline.back()->getOutputName(),
                                                                                                  source,
                                                                                                  intermediate,
                                                                                                  sink,
                                                                                                  additionalSources,
                                                                                                  shouldSwapLeftAndRight);

  // mark the state of this node as shuffled
  state = PDBJoinPhysicalNodeShuffled;

  // figure out if we have new sources
  std::list<PDBAbstractPhysicalNodePtr> newSources;
  if(otherSidePtr->state == PDBJoinPhysicalNodeShuffled) {
    newSources.insert(newSources.begin(), consumers.begin(), consumers.end());
  }

  // add all the consumed page sets
  std::list<PDBPageSetIdentifier> consumedPageSets = { source->pageSetIdentifier, intermediate->pageSetIdentifier };
  for(int i = 0; i < additionalSources->size(); ++i) {
    consumedPageSets.insert(consumedPageSets.begin(), (*additionalSources)[i]->pageSetIdentifier);
  }

  // set the page sets created, the produced page set has to have a page set
  std::vector<std::pair<PDBPageSetIdentifier, size_t>> newPageSets = { std::make_pair(sink->pageSetIdentifier, 1),
                                                                       std::make_pair(intermediate->pageSetIdentifier, 1) };

  // return the algorithm and the nodes that consume it's result
  return std::move(PDBPlanningResult(algorithm, newSources, consumedPageSets, newPageSets));
}

// set this value to some reasonable value // TODO this needs to be smarter
const size_t pdb::PDBJoinPhysicalNode::SHUFFLE_JOIN_THRASHOLD = std::numeric_limits<size_t>::max();


