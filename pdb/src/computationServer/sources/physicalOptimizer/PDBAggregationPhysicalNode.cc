//
// Created by dimitrije on 2/21/19.
//

#include <physicalAlgorithms/PDBAggregationPipeAlgorithm.h>
#include <physicalAlgorithms/PDBJoinAggregationAlgorithm.h>
#include <physicalOptimizer/PDBAggregationPhysicalNode.h>
#include <PDBSetObject.h>

#include <physicalOptimizer/PDBJoinPhysicalNode.h>
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

pdb::PDBPlanningResult PDBAggregationPhysicalNode::generateMergedAlgorithm(const PDBAbstractPhysicalNodePtr &lhs,
                                                                           const PDBAbstractPhysicalNodePtr &rhs,
                                                                           const PDBPageSetCosts &pageSetCosts) {

  // this page set is going to have the intermediate results of the LHS, the it is going to contain the JoinMap<hash, LHSKey>
  pdb::Handle<PDBSinkPageSetSpec> hashedLHSKey = pdb::makeObject<PDBSinkPageSetSpec>();
  hashedLHSKey->sinkType = PDBSinkType::HashedKeySink;
  hashedLHSKey->pageSetIdentifier = std::make_pair(computationID, (String) (pipeline.back()->getOutputName()  + "_hashed_lhs_key"));

  // this page set is going to have the intermediate results of the RHS, the it is going to contain the JoinMap<hash, RHSKey>
  pdb::Handle<PDBSinkPageSetSpec> hashedRHSKey = pdb::makeObject<PDBSinkPageSetSpec>();
  hashedRHSKey->sinkType = PDBSinkType::HashedKeySink;
  hashedRHSKey->pageSetIdentifier = std::make_pair(computationID, (String) (pipeline.back()->getOutputName() + "_hashed_rhs_key"));

  // this page set is going to have the intermediate results of the Aggregation Keys, the it is going to contain the JoinMap<AGG_TID, Vector<pair<LHS_TID, RHS_TID>>
  // there are also going to be two anonymous pages with Map<LHSKey, LHS_TID> and Map<RHSKey, RHS_Key>.
  pdb::Handle<PDBSinkPageSetSpec> aggregationTID = pdb::makeObject<PDBSinkPageSetSpec>();
  aggregationTID->sinkType = PDBSinkType::JoinAggregationTIDSink;
  aggregationTID->pageSetIdentifier = std::make_pair(computationID, (String) (pipeline.back()->getOutputName() + "_agg_tid"));

  // the join aggregation pipeline will end with end with an aggregation therefore the results will be in an aggregation sink
  pdb::Handle<PDBSinkPageSetSpec> sink = pdb::makeObject<PDBSinkPageSetSpec>();
  sink->sinkType = PDBSinkType::AggregationSink;
  sink->pageSetIdentifier = std::make_pair(computationID, (String) pipeline.back()->getOutputName());

  // the lhs key sink
  pdb::Handle<PDBSinkPageSetSpec> lhsKeySink = pdb::makeObject<PDBSinkPageSetSpec>();
  lhsKeySink->sinkType = PDBSinkType::HashedKeySink;
  lhsKeySink->pageSetIdentifier = PDBAbstractPageSet::toKeyPageSetIdentifier(std::make_pair(computationID,
                                                                                                      this->getPipeComputations().front()->getOutputName() + "_rhs"));
  // the rhs key sink
  pdb::Handle<PDBSinkPageSetSpec> rhsKeySink = pdb::makeObject<PDBSinkPageSetSpec>();
  rhsKeySink->sinkType = PDBSinkType::HashedKeySink;
  lhsKeySink->pageSetIdentifier = PDBAbstractPageSet::toKeyPageSetIdentifier(std::make_pair(computationID,
                                                                                                      this->getPipeComputations().front()->getOutputName() + "_lhs"));

  // the join aggregation key sink
  pdb::Handle<PDBSinkPageSetSpec> joinAggKeySink = pdb::makeObject<PDBSinkPageSetSpec>();
  joinAggKeySink->sinkType = PDBSinkType::JoinAggregationTIDSink;
  joinAggKeySink->pageSetIdentifier = PDBAbstractPageSet::toKeyPageSetIdentifier(std::make_pair(computationID,
                                                                                                          pipeline.back()->getOutputName() + "_join_agg"));

  // the left key source
  pdb::Handle<PDBSourcePageSetSpec> leftKeySource = pdb::makeObject<PDBSourcePageSetSpec>();
  leftKeySource->sourceType = PDBSourceType::SinglePageSource;
  leftKeySource->pageSetIdentifier = PDBAbstractPageSet::toKeyPageSetIdentifier(std::make_pair(computationID,
                                                                                                         this->getPipeComputations().front()->getOutputName() + "s_key_lhs"));

  // the right key source
  pdb::Handle<PDBSourcePageSetSpec> rightKeySource = pdb::makeObject<PDBSourcePageSetSpec>();
  rightKeySource->sourceType = PDBSourceType::SinglePageSource;
  rightKeySource->pageSetIdentifier = PDBAbstractPageSet::toKeyPageSetIdentifier(std::make_pair(computationID,
                                                                                                          this->getPipeComputations().front()->getOutputName() + "s_key_rhs"));

  // the plan source
  pdb::Handle<PDBSourcePageSetSpec> planSource = pdb::makeObject<PDBSourcePageSetSpec>();
  planSource->sourceType = PDBSourceType::SinglePageSource;
  planSource->pageSetIdentifier = PDBAbstractPageSet::toKeyPageSetIdentifier(std::make_pair(computationID,
                                                                                                      this->getPipeComputations().front()->getOutputName() + "s_key_plan"));

  // the right key source
  pdb::Handle<PDBSourcePageSetSpec> leftJoinSource = pdb::makeObject<PDBSourcePageSetSpec>();
  leftJoinSource->sourceType = PDBSourceType::ShuffledJoinTuplesSource;
  leftJoinSource->pageSetIdentifier = PDBAbstractPageSet::toKeyPageSetIdentifier(std::make_pair(computationID,
                                                                                                          this->getPipeComputations().front()->getOutputName() + "s_join_lhs"));

  // the plan source
  pdb::Handle<PDBSourcePageSetSpec> rightJoinSource = pdb::makeObject<PDBSourcePageSetSpec>();
  rightJoinSource->sourceType = PDBSourceType::ShuffledJoinTuplesSource;
  rightJoinSource->pageSetIdentifier = PDBAbstractPageSet::toKeyPageSetIdentifier(std::make_pair(computationID,
                                                                                                      this->getPipeComputations().front()->getOutputName() + "s_join_rhs"));
  // combine the sources from both pipelines
  auto additionalSources = lhs->getAdditionalSources();
  std::vector<pdb::Handle<PDBSourcePageSetSpec>> secondarySources;
  secondarySources.insert(secondarySources.end(), additionalSources.begin(), additionalSources.end());
  additionalSources = rhs->getAdditionalSources();
  secondarySources.insert(secondarySources.end(), additionalSources.begin(), additionalSources.end());

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

  // ok so we have to shuffle this side, generate the algorithm
  pdb::Handle<PDBJoinAggregationAlgorithm> algorithm = pdb::makeObject<PDBJoinAggregationAlgorithm>(lhs->getPrimarySources(),
                                                                                                    rhs->getPrimarySources(),
                                                                                                    sink,
                                                                                                    lhsKeySink,
                                                                                                    rhsKeySink,
                                                                                                    joinAggKeySink,
                                                                                                    leftKeySource,
                                                                                                    rightKeySource,
                                                                                                    leftJoinSource,
                                                                                                    rightJoinSource,
                                                                                                    planSource,
                                                                                                    lhs->getPipeComputations().front(),
                                                                                                    rhs->getPipeComputations().front(),
                                                                                                    this->getPipeComputations().front(),
                                                                                                    this->getPipeComputations().back(),
                                                                                                    hashedLHSKey,
                                                                                                    hashedRHSKey,
                                                                                                    aggregationTID,
                                                                                                    additionalSources,
                                                                                                    setsToMaterialize);


  std::list<PDBPageSetIdentifier> consumedPageSets = { hashedLHSKey->pageSetIdentifier,
                                                       hashedRHSKey->pageSetIdentifier,
                                                       lhsKeySink->pageSetIdentifier,
                                                       rhsKeySink->pageSetIdentifier,
                                                       leftKeySource->pageSetIdentifier,
                                                       rightKeySource->pageSetIdentifier,
                                                       leftJoinSource->pageSetIdentifier,
                                                       rightJoinSource->pageSetIdentifier,
                                                       planSource->pageSetIdentifier,
                                                       aggregationTID->pageSetIdentifier };

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
                                                                       std::make_pair(hashedLHSKey->pageSetIdentifier, 1),
                                                                       std::make_pair(hashedRHSKey->pageSetIdentifier, 1),
                                                                       std::make_pair(lhsKeySink->pageSetIdentifier, 1),
                                                                       std::make_pair(rhsKeySink->pageSetIdentifier, 1),
                                                                       std::make_pair(leftKeySource->pageSetIdentifier, 1),
                                                                       std::make_pair(rightKeySource->pageSetIdentifier, 1),
                                                                       std::make_pair(leftJoinSource->pageSetIdentifier, 1),
                                                                       std::make_pair(rightJoinSource->pageSetIdentifier, 1),
                                                                       std::make_pair(planSource->pageSetIdentifier, 1),
                                                                       std::make_pair(aggregationTID->pageSetIdentifier, 1) };


  // return the algorithm and the nodes that consume it's result
  return std::move(PDBPlanningResult(PDBPlanningResultType::GENERATED_ALGORITHM,
                                                  algorithm,consumers, consumedPageSets, newPageSets));
}

}