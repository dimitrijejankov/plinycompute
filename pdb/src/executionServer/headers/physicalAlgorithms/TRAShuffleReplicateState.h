#pragma once

#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBPageNetworkSender.h"
#include "PDBRandomAccessPageSet.h"
#include "TRAIndex.h"
#include "Pipeline.h"

namespace pdb {

struct TRAShuffleReplicateState : public PDBPhysicalAlgorithmState {

  PDBRandomAccessPageSetPtr inputPageSet = nullptr;

  TRAIndexNodePtr index = nullptr;

  TRAIndexNodePtr outputIndex = nullptr;

  pdb::PDBFeedingPageSetPtr feedingPageSet = nullptr;

  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> senders = nullptr;

  pdb::PDBPageSelfReceiverPtr selfReceiver = nullptr;

  std::shared_ptr<std::vector<PDBPageQueuePtr>> pageQueues = nullptr;

  pdb::PDBRandomAccessPageSetPtr outputPageSet = nullptr;
};

}

