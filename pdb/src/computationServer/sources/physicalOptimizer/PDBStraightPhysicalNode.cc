//
// Created by dimitrije on 2/21/19.
//

#include <physicalAlgorithms/PDBStraightPipeAlgorithm.h>
#include <physicalOptimizer/PDBStraightPhysicalNode.h>
#include <PDBVector.h>

PDBPipelineType pdb::PDBStraightPhysicalNode::getType() {
  return PDB_STRAIGHT_PIPELINE;
}

pdb::PDBPlanningResult pdb::PDBStraightPhysicalNode::generateAlgorithm(PDBAbstractPhysicalNodePtr &child,
                                                                       PDBPageSetCosts &pageSetCosts) {


  // can we pipeline this guy? we can do that if we only have one consumer
  if(consumers.size() == 1) {
    return consumers.front()->generatePipelinedAlgorithm(child, pageSetCosts);
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
  pdb::Handle<PDBStraightPipeAlgorithm> algorithm = pdb::makeObject<PDBStraightPipeAlgorithm>(primarySources,
                                                                                              pipeline.back(),
                                                                                              sink,
                                                                                              additionalSources,
                                                                                              setsToMaterialize);

  // add all the consumed page sets
  std::list<PDBPageSetIdentifier> consumedPageSets = {};
  for(auto &primarySource : primarySources) { consumedPageSets.insert(consumedPageSets.begin(), primarySource.source->pageSetIdentifier); }
  for(auto & additionalSource : additionalSources) { consumedPageSets.insert(consumedPageSets.begin(), additionalSource->pageSetIdentifier); }

  // set the page sets created
  std::vector<std::pair<PDBPageSetIdentifier, size_t>> newPageSets = { std::make_pair(sink->pageSetIdentifier, consumers.size()) };

  // return the algorithm and the nodes that consume it's result
  return std::move(PDBPlanningResult(PDBPlanningResultType::GENERATED_ALGORITHM, algorithm, consumers, consumedPageSets, newPageSets));
}