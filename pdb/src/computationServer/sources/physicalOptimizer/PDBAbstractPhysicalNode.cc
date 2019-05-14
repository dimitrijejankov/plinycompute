//
// Created by dimitrije on 2/21/19.
//

#include <PDBAbstractPhysicalNode.h>
#include <physicalOptimizer/PDBAbstractPhysicalNode.h>

pdb::PDBPlanningResult pdb::PDBAbstractPhysicalNode::generateAlgorithm(sourceCosts &sourcesWithIDs) {

  // create the additional sources vector, initially it is empty
  pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> additionalSources = makeObject<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>>();

  // this is the page set we are scanning
  pdb::Handle<PDBSourcePageSetSpec> source;

  // should we swap the left and right side if we have a join
  bool shouldSwapLeftAndRight = false;

  // are we doing a join
  if(isJoining()) {

    auto joinSources = getJoinSources(sourcesWithIDs);

    // add the right source to the additional sources
    additionalSources->push_back(std::get<1>(joinSources));

    // set the left source
    source = std::get<0>(joinSources);

    // should we swap left and right side of the join
    shouldSwapLeftAndRight = std::get<2>(joinSources);
  }
  else {

    // set the source set
    source = getSourcePageSet(sourcesWithIDs);
  }

  // generate the algorithm
  return generateAlgorithm(pipeline.front()->getOutputName(), source, sourcesWithIDs, additionalSources, shouldSwapLeftAndRight);
}

const std::list<pdb::PDBAbstractPhysicalNodePtr> pdb::PDBAbstractPhysicalNode::getProducers() {

  // create the list
  std::list<PDBAbstractPhysicalNodePtr> out;

  // fill up the list
  for(auto &it : producers) {
    out.push_back(it.lock());
  }

  // return the list
  return std::move(out);
}

const std::list<pdb::PDBAbstractPhysicalNodePtr> &pdb::PDBAbstractPhysicalNode::getConsumers() {
  return consumers;
}
