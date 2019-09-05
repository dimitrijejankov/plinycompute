#pragma once

// PRELOAD %PDBSourceSpec%

namespace pdb {


/**
 * This basically holds the info about the source of the pipeline
 * The tuple set it begins with, the actual set it scans if any or the page set that is scanned if any
 */
struct PDBSourceSpec : pdb::Object {

  ENABLE_DEEP_COPY

  /**
   * This is the tuple set of the atomic computation from which we are starting our pipeline
   */
  pdb::String firstTupleSet;

  /**
   * The source set we want to scan
   */
  pdb::Handle<PDBSetObject> sourceSet;

  /**
   * The source page set the algorithm should setup
   */
  pdb::Handle<PDBSourcePageSetSpec> pageSet;

  /**
   * Indicates whether the left and the right side are swapped
   */
  bool swapLHSandRHS = false;
};

}
