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
                                                                                PDBPageSetCosts &sourcesWithIDs,
                                                                                pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources,
                                                                                bool shouldSwapLeftAndRight) {

  // this is the same as @see generateAlgorithm except now the source is the source of the pipe we pipelined to this
  // and the additional source are transferred for that pipeline.
  return generateAlgorithm(startTupleSet, source, sourcesWithIDs, additionalSources, shouldSwapLeftAndRight);
}

pdb::PDBPlanningResult pdb::PDBStraightPhysicalNode::generateAlgorithm(const std::string &startTupleSet,
                                                                       const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                                       PDBPageSetCosts &sourcesWithIDs,
                                                                       pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &additionalSources,
                                                                       bool shouldSwapLeftAndRight) {


  // can we pipeline this guy? we can do that if we only have one consumer
  if(consumers.size() == 1) {
    return consumers.front()->generatePipelinedAlgorithm(startTupleSet, source, sourcesWithIDs, additionalSources, shouldSwapLeftAndRight);
  }

  // the sink is basically the last computation in the pipeline
  pdb::Handle<PDBSinkPageSetSpec> sink = pdb::makeObject<PDBSinkPageSetSpec>();
  sink->sinkType = PDBSinkType::SetSink;
  sink->pageSetIdentifier = std::make_pair(computationID, (String) pipeline.back()->getOutputName());

  // just store the sink page set for later use by the eventual consumers
  setSinkPageSet(sink);

  // figure out the materializations
  pdb::Handle<pdb::Vector<PDBSetObject>> setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();
  if(consumers.empty()) {

    // the last computation has to be a write set!
    if(pipeline.back()->getAtomicComputationTypeID() == WriteSetTypeID) {

      // cast the node to the output
      auto writerNode = std::dynamic_pointer_cast<WriteSet>(pipeline.back());

      // add the set of this node to the materialization
      setsToMaterialize->push_back(PDBSetObject(writerNode->getDBName(), writerNode->getSetName()));
    }
    else {

      // throw exception this is not supposed to happen
      throw runtime_error("TCAP does not end with a write set.");
    }
  }

  // generate the algorithm
  pdb::Handle<PDBStraightPipeAlgorithm> algorithm = pdb::makeObject<PDBStraightPipeAlgorithm>(startTupleSet,
                                                                                              pipeline.back()->getOutputName(),
                                                                                              source,
                                                                                              sink,
                                                                                              additionalSources,
                                                                                              setsToMaterialize,
                                                                                              shouldSwapLeftAndRight);

  // add all the consumed page sets
  std::list<PDBPageSetIdentifier> consumedPageSets = { source->pageSetIdentifier };
  for(int i = 0; i < additionalSources->size(); ++i) {
    consumedPageSets.insert(consumedPageSets.begin(), (*additionalSources)[i]->pageSetIdentifier);
  }

  // set the page sets created
  std::vector<std::pair<PDBPageSetIdentifier, size_t>> newPageSets = { std::make_pair(sink->pageSetIdentifier, consumers.size()) };

  // return the algorithm and the nodes that consume it's result
  return std::move(PDBPlanningResult(algorithm, consumers, consumedPageSets, newPageSets));
}