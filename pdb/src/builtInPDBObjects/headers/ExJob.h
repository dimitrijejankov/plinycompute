//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_PDBJOB_H
#define PDB_PDBJOB_H

#include <cstdint>
#include <PDBString.h>
#include <PDBVector.h>
#include <Computation.h>
#include "PDBPhysicalAlgorithm.h"

// PRELOAD %ExJob%

namespace pdb {

class ExJob : public Object  {
public:

  ENABLE_DEEP_COPY
  
  /**
   * The physical algorithm we want to run.
   */
  Handle<PDBPhysicalAlgorithm> physicalAlgorithm;

  /**
   * The computations we want to send
   */
  Handle<Vector<Handle<Computation>>> computations;

  /**
   * The tcap string of the computation
   */
  pdb::String tcap;

  /**
   * The id of the job
   */
  uint64_t jobID;

  /**
   * The id of the computation
   */
  uint64_t computationID;

  /**
   * The size of the computation
   */
  uint64_t computationSize;
};

}


#endif //PDB_PDBJOB_H
