//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_ExRunJob_H
#define PDB_ExRunJob_H

#include <cstdint>
#include <PDBString.h>
#include <PDBVector.h>
#include <Computation.h>
#include "PDBPhysicalAlgorithm.h"

// PRELOAD %ExRunJob%

namespace pdb {

class ExRunJob : public Object  {
public:

  ExRunJob() = default;

  ExRunJob(bool shouldRun) : shouldRun(shouldRun) {}

  bool shouldRun = true;

  ENABLE_DEEP_COPY
};

}


#endif //PDB_ExRunJob_H
