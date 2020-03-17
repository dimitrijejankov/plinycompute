#pragma once

#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBPageNetworkSender.h"
#include "EightWayJoinPipeline.h"
#include "Pipeline.h"

namespace pdb {

struct PDBJoin8AlgorithmState : public PDBPhysicalAlgorithmState {

  // the page sets we get the keys from
  std::unordered_map<int32_t , PDBAbstractPageSetPtr> keySourcePageSets;

  std::shared_ptr<EightWayJoinPipeline> keyPipeline;

  //
  std::shared_ptr<std::vector<PDBPageQueuePtr>> planPageQueues = nullptr;

  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> planSenders;

  pdb::PDBFeedingPageSetPtr planPageSet = nullptr;

  PDBPageHandle planPage;
};

}