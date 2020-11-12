#pragma once

#include <PDBRandomAccessPageSet.h>
#include <TRAIndex.h>
#include <TRABlock.h>
#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBPageNetworkSender.h"
#include "Pipeline.h"

namespace pdb {

struct MM3DState : public PDBPhysicalAlgorithmState {

  pdb::PDBRandomAccessPageSetPtr lhsSet = nullptr;

  pdb::PDBRandomAccessPageSetPtr rhsSet = nullptr;

  std::shared_ptr<std::vector<PDBPageQueuePtr>> pageQueuesLHS = nullptr;
  std::shared_ptr<std::vector<PDBPageQueuePtr>> pageQueuesRHS = nullptr;
  std::shared_ptr<std::vector<PDBPageQueuePtr>> outQueues = nullptr;

  pdb::PDBPageSelfReceiverPtr selfReceiverLHS = nullptr;
  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> sendersLHS = nullptr;

  pdb::PDBPageSelfReceiverPtr selfReceiverRHS = nullptr;
  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> sendersRHS = nullptr;

  pdb::PDBFeedingPageSetPtr feedingPageSetLHS = nullptr;
  pdb::PDBFeedingPageSetPtr feedingPageSetRHS = nullptr;

  map<tuple<int32_t, int32_t>, TRABlock*> indexLHS;
  map<tuple<int32_t, int32_t>, TRABlock*> indexRHS;
};

}