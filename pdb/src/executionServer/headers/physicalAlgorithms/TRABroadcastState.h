#pragma once

#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBPageNetworkSender.h"
#include "Pipeline.h"
#include "PDBRandomAccessPageSet.h"
#include "TRAIndex.h"

namespace pdb {

struct TRABroadcastState : public PDBPhysicalAlgorithmState {


  pdb::PDBPageSelfReceiverPtr selfReceiver = nullptr;

  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> senders = nullptr;

  std::shared_ptr<std::vector<PDBPageQueuePtr>> pageQueues = nullptr;

  PDBAbstractPageSetPtr inputSet = nullptr;

  pdb::PDBFeedingPageSetPtr feedingPageSet = nullptr;

  PDBRandomAccessPageSetPtr indexedPageSet = nullptr;

  TRAIndexNodePtr index = nullptr;
};

}

