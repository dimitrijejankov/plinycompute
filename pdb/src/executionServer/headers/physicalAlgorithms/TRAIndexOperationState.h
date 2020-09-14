#pragma once

#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBPageNetworkSender.h"
#include "Pipeline.h"
#include "PDBRandomAccessPageSet.h"
#include "TRAIndex.h"

namespace pdb {

struct TRAIndexOperationState : public PDBPhysicalAlgorithmState {

  PDBRandomAccessPageSetPtr inputSet = nullptr;

  TRAIndexNodePtr index = nullptr;

};

}

