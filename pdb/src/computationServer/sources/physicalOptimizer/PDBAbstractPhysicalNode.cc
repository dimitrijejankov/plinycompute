//
// Created by dimitrije on 2/21/19.
//

#include <PDBAbstractPhysicalNode.h>
#include <physicalOptimizer/PDBAbstractPhysicalNode.h>

pdb::PDBPlanningResult pdb::PDBAbstractPhysicalNode::generateAlgorithm(PDBPageSetCosts &pageSetCosts) {

  // this is the page set we are scanning
  pdb::Handle<PDBSourcePageSetSpec> source;

  // should we swap the left and right side if we have a join
  bool shouldSwapLeftAndRight = false;

  // are we doing a join
  if(isJoining()) {

    auto joinSources = getJoinSources(pageSetCosts);

    // add the right source to the additional sources
    additionalSources.push_back(std::get<1>(joinSources));

    // set the left source
    source = std::get<0>(joinSources);

    // should we swap left and right side of the join
    shouldSwapLeftAndRight = std::get<2>(joinSources);
  }
  else {

    // set the source set
    source = getSourcePageSet(pageSetCosts);
  }

  // create a new pipeline plan
  primarySources.emplace_back();

  // set the pipeline plan for this
  auto &plannedPipeline = primarySources.back();
  plannedPipeline.source = source;
  plannedPipeline.startAtomicComputation = pipeline.front();
  plannedPipeline.shouldSwapLeftAndRight = shouldSwapLeftAndRight;

  // generate the algorithm
  auto myHandle = getHandle();
  return generateAlgorithm(myHandle, pageSetCosts);
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

pdb::PDBPlanningResult pdb::PDBAbstractPhysicalNode::generatePipelinedAlgorithm(PDBAbstractPhysicalNodePtr &child,
                                                                                PDBPageSetCosts &sourcesWithIDs) {

  // copy the additional sources from the child
  additionalSources = child->additionalSources;

  // check if we are doing an union here and if we haven't processed the left side already, this is the left side
  if(isUnioning() && primarySources.empty()) {

    // set all the unions we got from the
    primarySources.insert(primarySources.end(), child->primarySources.begin(), child->primarySources.end());

    // copy all the sources consumed by the other pipelines
    std::list<PDBPageSetIdentifier> consumedSources;
    for(auto &p : child->primarySources) {
      consumedSources.emplace_back(p.source->pageSetIdentifier);
    }

    // inform the physical planner that we have consumed the source and that it has been pipelined into the union
    return std::move(PDBPlanningResult(PDBPlanningResultType::UNIONED_PIPELINE, nullptr, {}, consumedSources, {}));
  }

  // just copy the pipelines from the child
  primarySources.insert(primarySources.end(), child->primarySources.begin(), child->primarySources.end());

  // this is the same as @see generateAlgorithm except now the source is the source of the pipe we pipelined to this
  // and the additional source are transferred for that pipeline. We can not pipeline an aggregation
  auto me = getHandle();
  return std::move(generateAlgorithm(me, sourcesWithIDs));
}
