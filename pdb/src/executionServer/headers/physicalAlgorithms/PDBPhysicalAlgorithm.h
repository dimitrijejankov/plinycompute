//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_PDBPHYSICALALGORITHM_H
#define PDB_PDBPHYSICALALGORITHM_H

#include <Object.h>
#include <PDBString.h>
#include <PDBSourcePageSetSpec.h>
#include <PDBSinkPageSetSpec.h>

namespace pdb {

enum PDBAbstractAlgorithmType {

  ShuffleForJoin,
  BroadcastForJoin,
  DistributedAggregation,
  StraightPipe
};


class PDBPhysicalAlgorithm : public Object {
public:

  /**
   * Sets up the whole algorithm
   */
  virtual void setup() = 0;

  /**
   * Runs the algorithm
   */
  virtual void run() = 0;

  /**
   * Returns the type of the algorithm we want to run
   */
  virtual PDBAbstractAlgorithmType getAlgorithmType() = 0;

private:

  /**
   * The source type the algorithm should setup
   */
  PDBSourcePageSetSpec source;

  /**
   * The sink type the algorithm should setup
   */
  PDBSinkPageSetSpec sink;

  /**
   * List of secondary sources like hash sets for join etc..
   */
  std::vector<PDBSourcePageSetSpec> secondarySources;

};

}

#endif //PDB_PDBPHYSICALALGORITHM_H
