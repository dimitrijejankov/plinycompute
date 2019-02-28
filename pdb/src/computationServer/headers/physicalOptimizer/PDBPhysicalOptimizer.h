/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/
#ifndef PDB_PHYSICAL_OPTIMIZER_H
#define PDB_PHYSICAL_OPTIMIZER_H


#include "Computation.h"
#include "PDBVector.h"
#include <PDBLogger.h>
#include <AtomicComputationList.h>
#include <PDBPhysicalAlgorithm.h>
#include <PDBPipeNodeBuilder.h>

namespace pdb {

/**
 * This class basically takes in a TCAP and breaks it up into PhysicalAlgorithms, that are going to be sent to,
 * the @see ExecutionServerFrontend
 */
class PDBPhysicalOptimizer {
public:

  /**
   * Takes in the TCAP string that we want to analyze and to the physical optimization on
   * @param tcapString - the TACP string
   * @param logger - the logger
   */
  PDBPhysicalOptimizer(String tcapString, PDBLoggerPtr &logger);

  /**
   * Default destructor
   */
  ~PDBPhysicalOptimizer() = default;

  /**
   * Returns the next algorithm we want to run.
   * Warning this assumes a that an allocating block has been setup previously
   * @return the algorithm - the algorithm we want to run
   */
  pdb::Handle<pdb::PDBPhysicalAlgorithm> getNextAlgorithm();

  /**
   * This returns true if there is an algorithm we need to run in order to finish the computation
   * @return true if there is false otherwise
   */
  bool hasAlgorithmToRun();

  /**
   * Updates the set statistics
   */
  void updateStats();

private:

  /**
   * These are all the sources we currently have
   */
  std::vector<pdb::PDBAbstractPhysicalNodePtr> sources;

  /**
   * The logger associated with the physical optimizer
   */
  PDBLoggerPtr logger;

};

}


#endif //PDB_PHYSICAL_OPTIMIZER_H
