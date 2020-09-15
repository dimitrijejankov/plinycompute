#pragma once

#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBPageNetworkSender.h"
#include "PDBRandomAccessPageSet.h"
#include "TRAIndex.h"
#include "Pipeline.h"

namespace pdb {

struct TRALocalAggregationState : public PDBPhysicalAlgorithmState {

  pdb::PDBRandomAccessPageSetPtr inputSet;

  // get the rhs page set
  pdb::PDBRandomAccessPageSetPtr outputSet;

  // the index
  TRAIndexNodePtr index = nullptr;
  TRAIndexNodePtr outputIndex = nullptr;
};

}

