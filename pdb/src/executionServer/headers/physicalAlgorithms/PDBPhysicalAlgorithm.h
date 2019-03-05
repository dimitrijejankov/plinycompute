#include <utility>

//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_PDBPHYSICALALGORITHM_H
#define PDB_PDBPHYSICALALGORITHM_H

#include <Object.h>
#include <PDBString.h>
#include <PDBSourcePageSetSpec.h>
#include <PDBSinkPageSetSpec.h>
#include <PDBVector.h>

namespace pdb {

enum PDBPhysicalAlgorithmType {

  ShuffleForJoin,
  BroadcastForJoin,
  DistributedAggregation,
  StraightPipe
};


class PDBPhysicalAlgorithm : public Object {
public:

  PDBPhysicalAlgorithm() = default;

  ~PDBPhysicalAlgorithm() = default;

  PDBPhysicalAlgorithm(const pdb::Handle<PDBSourcePageSetSpec> &source,
                       const pdb::Handle<PDBSinkPageSetSpec> &sink,
                       const pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> &secondarySources)
      : source(source), sink(sink), secondarySources(secondarySources) {}

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
  virtual PDBPhysicalAlgorithmType getAlgorithmType() = 0;

protected:

  /**
   * The source type the algorithm should setup
   */
  pdb::Handle<PDBSourcePageSetSpec> source;

  /**
   * The sink type the algorithm should setup
   */
  pdb::Handle<PDBSinkPageSetSpec> sink;

  /**
   * List of secondary sources like hash sets for join etc.. null if there are no secondary sources
   */
  pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> secondarySources;

};

}

#endif //PDB_PDBPHYSICALALGORITHM_H
