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

namespace pdb {

class PDBPhysicalOptimizer {
public:

  PDBPhysicalOptimizer(String tcapString, PDBLoggerPtr &logger);

  ~PDBPhysicalOptimizer() = default;

 private:

  std::unique_ptr<AtomicComputationList> atomicComputations;

  PDBLoggerPtr logger;

};

}


#endif //PDB_PHYSICAL_OPTIMIZER_H
