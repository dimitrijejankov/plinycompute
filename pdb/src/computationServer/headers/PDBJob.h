//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_PDBJOB_H
#define PDB_PDBJOB_H

#include <cstdint>
#include <PDBString.h>
#include <PDBVector.h>
#include <Computation.h>
#include "PDBAbstractAlgorithm.h"

namespace pdb {

class PDBJob { // Shouldn't this inherit from pdb::Object?
public:

  /**
   * The physical algorithm we want to run.
   */
  Handle<PDBAbstractAlgorithm> physicalAlgorithm;

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
};

}


#endif //PDB_PDBJOB_H
