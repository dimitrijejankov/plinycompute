#pragma once

#include <PDBPageSelfReceiver.h>
#include "PDBPhysicalAlgorithmState.h"
#include "PDBLabeledPageSet.h"
#include "PDBFeedingPageSet.h"
#include "PDBPageNetworkSender.h"
#include "JoinAggSideSender.h"
#include "JoinAggTupleEmitter.h"
#include "JoinMapCreator.h"
#include "Pipeline.h"

namespace pdb {

struct PDBJoinAggregationState : public PDBPhysicalAlgorithmState {


  /**
   * The labled left page set of keys
   */
  pdb::PDBLabeledPageSetPtr labeledLeftPageSet = nullptr;

  /**
   * The labled right page set of keys
   */
  pdb::PDBLabeledPageSetPtr labeledRightPageSet = nullptr;

  /**
   *
   */
  pdb::PDBAnonymousPageSetPtr joinAggPageSet = nullptr;

  /**
   *
   */
  pdb::PDBRandomAccessPageSetPtr leftShuffledPageSet = nullptr;

  /**
   *
   */
  pdb::PDBRandomAccessPageSetPtr rightShuffledPageSet = nullptr;

  /**
   *
   */
  pdb::PDBAnonymousPageSetPtr intermediatePageSet = nullptr;

  /**
   *
   */
  pdb::PDBFeedingPageSetPtr preaggPageSet = nullptr;

  /**
   *
   */
  PDBPageHandle leftKeyPage = nullptr;

  /**
   *
   */
  PDBPageHandle rightKeyPage = nullptr;

  /**
   *
   */
  PDBPageHandle aggKeyPage = nullptr;

  /**
   *
   */
  PDBPageHandle planPage;

  /**
   *
   */
  std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> leftTIDToRecordMapping;

  /**
   *
   */
  std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> rightTIDToRecordMapping;

  /**
   * Connections to nodes (including self connection to this one) for the left side of the join
   */
  std::shared_ptr<std::vector<PDBCommunicatorPtr>> leftJoinSideCommunicatorsOut = nullptr;

  /**
   * Connections to nodes (including self connection to this one) for the right side of the join
   */
  std::shared_ptr<std::vector<PDBCommunicatorPtr>> rightJoinSideCommunicatorsOut = nullptr;

  /**
   * Connections to nodes (including self connection to this one) for the left side of the join
   */
  std::shared_ptr<std::vector<PDBCommunicatorPtr>> leftJoinSideCommunicatorsIn = nullptr;

  /**
   * Connections to nodes (including self connection to this one) for the right side of the join
   */
  std::shared_ptr<std::vector<PDBCommunicatorPtr>> rightJoinSideCommunicatorsIn = nullptr;

  /**
   * These send records to the nodes for the left side
   */
  std::shared_ptr<std::vector<JoinAggSideSenderPtr>> leftJoinSideSenders = nullptr;

  /**
   * These send records to the nodes for the right side
   */
  std::shared_ptr<std::vector<JoinAggSideSenderPtr>> rightJoinSideSenders = nullptr;

  /**
   * This goes receives join records with their TID, goes through them and keeps track on what page they are.
   */
  std::shared_ptr<std::vector<JoinMapCreatorPtr>> rightJoinMapCreators = nullptr;

  /**
   * This goes receives join records with their TID, goes through them and keeps track on what page they are.
   */
  std::shared_ptr<std::vector<JoinMapCreatorPtr>> leftJoinMapCreators = nullptr;

  /**
   * The join key side pipelines
   */
  std::shared_ptr<std::vector<PipelinePtr>> joinKeyPipelines = nullptr;

  /**
   * The preaggregation pipelines
   */
  std::shared_ptr<std::vector<PipelinePtr>> preaggregationPipelines = nullptr;

  /**
   * This forwards the preaggregated pages to this node
   */
  pdb::PDBPageSelfReceiverPtr selfReceiver;

  /**
   * These senders forward pages that are for other nodes
   */
  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> senders;

  /**
   * The aggregation pipelines
   */
  std::shared_ptr<std::vector<PipelinePtr>> aggregationPipelines = nullptr;

  /**
   *
   */
  std::shared_ptr<std::vector<PDBPageQueuePtr>> pageQueues = nullptr;

  /**
   * The join aggregation pipeline
   */
  PipelinePtr joinKeyAggPipeline = nullptr;

  /**
   * This runs the left and right side of the join
   */
  std::shared_ptr<std::vector<PipelinePtr>> joinPipelines = nullptr;

  /**
   * This is used to emit join tuples
   */
  std::shared_ptr<JoinAggTupleEmitter> emitter = nullptr;

  /**
   * The key extractor
   */
  PDBKeyExtractorPtr keyExtractor;

  /**
   * Is a local aggregation enough
   */
  bool localAggregation = false;

  /**
   * The left and right join side task
   */
  static const int32_t LEFT_JOIN_SIDE_TASK;
  static const int32_t RIGHT_JOIN_SIDE_TASK;
  static const int32_t PLAN_TASK;
};

}
