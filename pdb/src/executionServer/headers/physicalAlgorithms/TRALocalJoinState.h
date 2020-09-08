#pragma once

#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBPageNetworkSender.h"
#include "Pipeline.h"
#include "PDBSetPageSet.h"
#include "PDBRandomAccessPageSet.h"

namespace pdb {

struct TRALocalJoinState : public PDBPhysicalAlgorithmState {


  // the the input page set
  pdb::PDBSetPageSetPtr inputPageSet;

  // the emmitter will put set pageser here
  pdb::PDBRandomAccessPageSetPtr leftPageSet;

  // get the rhs page set
  pdb::PDBAbstractPageSetPtr rightPageSet;

  // get the in
  pdb::PDBAnonymousPageSetPtr output;
};

}

