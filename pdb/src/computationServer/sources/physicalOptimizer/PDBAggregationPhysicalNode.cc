//
// Created by dimitrije on 2/21/19.
//

#include <physicalAlgorithms/PDBAggregationPipeAlgorithm.h>
#include <physicalOptimizer/PDBAggregationPhysicalNode.h>
#include <PDBSetObject.h>

#include "physicalOptimizer/PDBAggregationPhysicalNode.h"


namespace pdb {

PDBPipelineType pdb::PDBAggregationPhysicalNode::getType() {
  return PDB_AGGREGATION_PIPELINE;
}

pdb::PDBPlanningResult PDBAggregationPhysicalNode::generateAlgorithm(PDBAbstractPhysicalNodePtr &child,
                                                                     PDBPageSetCosts &pageSetCosts) {

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

  // figure out if we need to materialize the result of the aggregation, this happens if the aggregation is directly added to a write set
  // in that case the next node (it's consumer) will contain two atomic computations, one that starts at the aggregation, the other that is a
  // write set, there fore we check for exactly that and if we find it we need to materialize
  pdb::Handle<pdb::Vector<PDBSetObject>> setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();
  for(auto consumer = consumers.begin(); consumer != consumers.end();) {

    // if it only has two computations in the pipeline, mark that we need to materialize the result
    auto &computations = (*consumer)->getPipeComputations();
    if(computations.size() == 2 && computations[0]->getAtomicComputationTypeID() == ApplyAggTypeID &&
                                   computations[1]->getAtomicComputationTypeID() == WriteSetTypeID) {

      // cast the node to the output
      auto writerNode = std::dynamic_pointer_cast<WriteSet>(computations[1]);

      // add the set of this node to the materialization
      setsToMaterialize->push_back(PDBSetObject(writerNode->getDBName(), writerNode->getSetName()));

      // remove this consumer
      auto tmp = consumer++;
      removeConsumer(*tmp);
    }

    // go to the next one
    consumer++;
  }

  // just store the sink page set for later use by the eventual consumers
  setSinkPageSet(sink);

  // create the algorithm
  pdb::Handle<PDBAggregationPipeAlgorithm> algorithm = pdb::makeObject<PDBAggregationPipeAlgorithm>(primarySources,
                                                                                                    pipeline.back(),
                                                                                                    hashedToSend,
                                                                                                    hashedToRecv,
                                                                                                    sink,
                                                                                                    additionalSources,
                                                                                                    setsToMaterialize);
  // add all the consumed page sets
  std::list<PDBPageSetIdentifier> consumedPageSets = { hashedToSend->pageSetIdentifier, hashedToRecv->pageSetIdentifier };
  for(auto &primarySource : primarySources) { consumedPageSets.insert(consumedPageSets.begin(), primarySource.source->pageSetIdentifier); }
  for(auto & additionalSource : additionalSources) { consumedPageSets.insert(consumedPageSets.begin(), additionalSource->pageSetIdentifier); }

  // if there are no consumers, (this happens if all the consumers are materializations), mark the ink set as consumed too
  size_t sinkConsumers = consumers.size();
  if(consumers.empty()) {

    // since we are materializing this set we are kind of consuming it
    sinkConsumers = 1;

    // add the sink to the list of consumed page sets
    consumedPageSets.insert(consumedPageSets.begin(), sink->pageSetIdentifier);
  }

  // set the page sets created
  std::vector<std::pair<PDBPageSetIdentifier, size_t>> newPageSets = { std::make_pair(sink->pageSetIdentifier, sinkConsumers),
                                                                       std::make_pair(hashedToSend->pageSetIdentifier, 1),
                                                                       std::make_pair(hashedToRecv->pageSetIdentifier, 1) };

  // return the algorithm and the nodes that consume it's result
  return std::move(PDBPlanningResult(PDBPlanningResultType::GENERATED_ALGORITHM, algorithm, consumers, consumedPageSets, newPageSets));
}

}