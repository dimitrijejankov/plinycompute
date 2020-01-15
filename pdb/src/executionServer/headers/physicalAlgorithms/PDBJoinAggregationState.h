#pragma once

#include "PDBPhysicalAlgorithmState.h"
#include "PDBLabeledPageSet.h"
#include "PDBFeedingPageSet.h"
#include "PDBPageNetworkSender.h"
#include "JoinAggSideSender.h"
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
  pdb::PDBAnonymousPageSetPtr leftShuffledPageSet = nullptr;

  /**
   *
   */
  pdb::PDBAnonymousPageSetPtr rightShuffledPageSet = nullptr;

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
  pdb::PDBFeedingPageSetPtr leftKeyToNodePageSet = nullptr;

  /**
   *
   */
  pdb::PDBFeedingPageSetPtr rightKeyToNodePageSet= nullptr;

  /**
   *
   */
  pdb::PDBFeedingPageSetPtr planPageSet = nullptr;

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
  PDBPageHandle planPage;

  /**
   *
   */
  std::shared_ptr<std::vector<PDBPageQueuePtr>> leftKeyPageQueues = nullptr;

  /**
   *
   */
  std::shared_ptr<std::vector<PDBPageQueuePtr>> rightKeyPageQueues = nullptr;

  /**
   *
   */
  std::shared_ptr<std::vector<PDBPageQueuePtr>> planPageQueues = nullptr;

  /**
   *
   */
  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> leftKeySenders;

  /**
   *
   */
  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> rightKeySenders;

  /**
   * This sends the plan
   */
  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> planSenders;

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
   * This takes in the records from the side of the join and makes them into a tuple set
   */
  std::shared_ptr<std::vector<JoinMapCreatorPtr>> joinMapCreators = nullptr;

  /**
   * The join key side pipelines
   */
  std::shared_ptr<std::vector<PipelinePtr>> joinKeyPipelines = nullptr;

  /**
   * The preaggregation pipelines
   */
  std::shared_ptr<std::vector<PipelinePtr>> preaggregationPipelines = nullptr;

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
   * The left and right join side task
   */
  static const int32_t LEFT_JOIN_SIDE_TASK;
  static const int32_t RIGHT_JOIN_SIDE_TASK;
};

}
