//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_STRAIGHTPIPEALGORITHM_H
#define PDB_STRAIGHTPIPEALGORITHM_H

#include "PDBPhysicalAlgorithm.h"

/**
 * This is important do not remove, it is used by the generator
 */
// PRELOAD %PDBStraightPipeAlgorithm%


namespace pdb {

class PDBStraightPipeAlgorithm : public PDBPhysicalAlgorithm {
public:

  /**
   * //TODO
   */
  void setup() override;

  /**
   * //TODO
   */
  void run() override;

  /**
   * Returns StraightPipe as the type
   * @return the type
   */
  PDBAbstractAlgorithmType getAlgorithmType() override;

};

}


#endif //PDB_STRAIGHTPIPEALGORITHM_H
