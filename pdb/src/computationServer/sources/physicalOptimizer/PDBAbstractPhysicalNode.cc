//
// Created by dimitrije on 2/21/19.
//

#include <PDBAbstractPhysicalNode.h>
#include <physicalOptimizer/PDBAbstractPhysicalNode.h>

pdb::PDBPlanningResult pdb::PDBAbstractPhysicalNode::generateAlgorithm(sourceCosts &sourcesWithIDs) {

  // create the additional sources vector, initially it is empty
  pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> additionalSources = makeObject<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>>();

  // are we doing a join
  if(isJoining()) {

    // add the right source to the additional sources
    additionalSources->push_back(getRightSourcePageSet(sourcesWithIDs));
  }

  // this is the page set we are scanning
  pdb::Handle<PDBSourcePageSetSpec> source = getSourcePageSet(sourcesWithIDs);

  // generate the algorithm
  return generateAlgorithm(pipeline.front()->getOutputName(), source, sourcesWithIDs, additionalSources);
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
