#pragma once


#include "Computation.h"
#include "PDBVector.h"
#include <PDBLogger.h>
#include <AtomicComputationList.h>
#include <PDBPhysicalAlgorithm.h>
#include <PDBPipeNodeBuilder.h>
#include <PDBDistributedStorage.h>
#include "PDBOptimizerSource.h"

namespace pdb {

/**
 * This class basically takes in a TCAP and breaks it up into PhysicalAlgorithms, that are going to be sent to,
 * the @see ExecutionServerFrontend
 */
class HardCodedOptimizer {
 public:

  /**
   * Takes in the TCAP string that we want to analyze and to the physical optimization on
   * @param computationID - the id of the computation this optimizer is to optimize
   * @param tcapString - the TACP string
   * @param keyedComputations - the indicates what computations are keyed, maps the name of the computation
   * @param clientPtr - the catalog client
   * @param logger - the logger
   */
  HardCodedOptimizer(uint64_t computationID);

  pdb::Handle<pdb::PDBPhysicalAlgorithm> getNextAlgorithm();

  bool hasAlgorithmToRun();

  std::vector<PDBPageSetIdentifier> getPageSetsToRemove();

  uint64_t computationID;
};
}