#pragma once

#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBPageNetworkSender.h"
#include "Pipeline.h"
#include "PDBSetPageSet.h"
#include "TRAIndex.h"
#include "PDBRandomAccessPageSet.h"
#include "TRALocalJoinEmitter.h"

namespace pdb {

struct TRALocalJoinState : public PDBPhysicalAlgorithmState {


  // the the input page set
  pdb::PDBSetPageSetPtr inputPageSet;

  // the emmitter will put set pageset here
  pdb::PDBRandomAccessPageSetPtr leftPageSet;

  // get the rhs page set
  pdb::PDBRandomAccessPageSetPtr rightPageSet;

  // get the in
  pdb::PDBAnonymousPageSetPtr output;

  // the index
  TRAIndexNodePtr index = nullptr;

  // the join record emitter
  std::shared_ptr<TRALocalJoinEmitter> emitter;
};

}

