#pragma once

#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBAggregationPipeState.h"
#include "PDBPageNetworkSender.h"
#include "Pipeline.h"

namespace pdb {

struct PDBShuffleForJoinState : public PDBPhysicalAlgorithmState {

  /**
   * This forwards the preaggregated pages to this node
   */
  pdb::PDBPageSelfReceiverPtr selfReceiver = nullptr;

  /**
   * These senders forward pages that are for other nodes
   */
  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> senders = nullptr;

  /**
   * The pipelines that run the shuffling
   */
  std::shared_ptr<std::vector<PipelinePtr>> joinShufflePipelines = nullptr;

  /**
   * The page queues
   */
  std::shared_ptr<std::vector<PDBPageQueuePtr>> pageQueues = nullptr;

};

}