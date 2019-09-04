#pragma once

#include <AtomicComputation.h>
#include <PDBSourcePageSetSpec.h>
#include <Handle.h>
#include <PDBVector.h>

namespace pdb {

struct PDBPrimarySource {

  /**
   * The starting atomic computation of the left pipeline
   */
  AtomicComputationPtr startAtomicComputation = nullptr;

  /**
   * The source of the left pipeline
   */
  pdb::Handle<PDBSourcePageSetSpec> source = nullptr;

  /**
   * True if this pipeline needs to swap the left and the right pipeline for the join source
   */
  bool shouldSwapLeftAndRight = false;

};

}
