#pragma once

#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBPageNetworkSender.h"
#include "Pipeline.h"
#include "PDBRandomAccessPageSet.h"

namespace pdb {

struct TRAMaterializeOperationState : public PDBPhysicalAlgorithmState {

  PDBRandomAccessPageSetPtr inputSet = nullptr;

};

}

