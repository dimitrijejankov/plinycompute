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

// PRELOAD %PDBPhysicalAlgorithm%

class PDBPhysicalAlgorithm : public Object {
public:

  ENABLE_DEEP_COPY

  PDBPhysicalAlgorithm() = default;

  ~PDBPhysicalAlgorithm() = default;

  PDBPhysicalAlgorithm(const pdb::Handle<PDBSourcePageSetSpec> &source,
                       const pdb::Handle<PDBSinkPageSetSpec> &sink,
                       const pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> &secondarySources)
      : source(source), sink(sink), secondarySources(secondarySources) {}

  /**
   * Sets up the whole algorithm
   */
  virtual void setup() { throw std::runtime_error("Can not setup PDBPhysicalAlgorithm that is an abstract class"); };

  /**
   * Runs the algorithm
   */
  virtual void run() { throw std::runtime_error("Can not run PDBPhysicalAlgorithm that is an abstract class"); };

  /**
   * Returns the type of the algorithm we want to run
   */
  virtual PDBPhysicalAlgorithmType getAlgorithmType() { throw std::runtime_error("Can not get the type of the base class"); };

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
