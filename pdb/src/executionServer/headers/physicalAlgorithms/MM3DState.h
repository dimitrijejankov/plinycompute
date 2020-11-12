#pragma once

#include <PDBRandomAccessPageSet.h>
#include <TRAIndex.h>
#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBPageNetworkSender.h"
#include "Pipeline.h"

namespace pdb {

struct MM3DState : public PDBPhysicalAlgorithmState {

  pdb::PDBRandomAccessPageSetPtr lhsSet = nullptr;

  pdb::PDBRandomAccessPageSetPtr rhsSet = nullptr;

  std::vector<void*> lhsReceived;

  std::vector<void*> rhsReceived;

  std::shared_ptr<std::vector<PDBPageQueuePtr>> pageQueuesLHS = nullptr;
  std::shared_ptr<std::vector<PDBPageQueuePtr>> pageQueuesRHS = nullptr;

  pdb::PDBPageSelfReceiverPtr selfReceiverLHS = nullptr;
  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> sendersLHS = nullptr;

  pdb::PDBPageSelfReceiverPtr selfReceiverRHS = nullptr;
  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> sendersRHS = nullptr;

  pdb::PDBFeedingPageSetPtr feedingPageSetLHS = nullptr;
  pdb::PDBFeedingPageSetPtr feedingPageSetRHS = nullptr;
};

}