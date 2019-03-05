//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_STRAIGHTPIPEALGORITHM_H
#define PDB_STRAIGHTPIPEALGORITHM_H

#include "PDBPhysicalAlgorithm.h"

/**
 * This is important do not remove, it is used by the generator
 */

namespace pdb {

// PRELOAD %PDBStraightPipeAlgorithm%

class PDBStraightPipeAlgorithm : public PDBPhysicalAlgorithm {
public:

  ENABLE_DEEP_COPY

  PDBStraightPipeAlgorithm() = default;
  ~PDBStraightPipeAlgorithm() = default;


  PDBStraightPipeAlgorithm(const pdb::Handle<PDBSourcePageSetSpec> &source,
                           const pdb::Handle<PDBSinkPageSetSpec> &sink,
                           const pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> &secondarySources);

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
  PDBPhysicalAlgorithmType getAlgorithmType() override;

};

}


#endif //PDB_STRAIGHTPIPEALGORITHM_H
