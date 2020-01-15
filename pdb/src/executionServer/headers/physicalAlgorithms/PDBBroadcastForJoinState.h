#pragma once

#include <PDBPageNetworkSender.h>
#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "Pipeline.h"
#include "LogicalPlan.h"

namespace pdb {

struct PDBBroadcastForJoinState : public PDBPhysicalAlgorithmState {

  /**
   * This forwards the preaggregated pages to this node
   */
  pdb::PDBPageSelfReceiverPtr selfReceiver;

  /**
   * These senders forward pages that are for other nodes
   */
  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> senders;


  /**
   * Vector of pipelines that will run this algorithm. The pipelines will be built when you call setup on this object.
   * This must be null when sending this object.
   */
  std::shared_ptr<std::vector<PipelinePtr>> prebroadcastjoinPipelines = nullptr;

  /**
   *
   */
  std::shared_ptr<std::vector<PipelinePtr>> broadcastjoinPipelines = nullptr;

  /**
   *
   */
  std::shared_ptr<std::vector<PDBPageQueuePtr>> pageQueues = nullptr;

  /**
   *
   */
  LogicalPlanPtr logicalPlan;

};

}

