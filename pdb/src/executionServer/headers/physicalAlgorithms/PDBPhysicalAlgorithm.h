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
#include <gtest/gtest_prod.h>

namespace pdb {

// predefine this so avoid recursive definition
class ExJob;
class PDBStorageManagerBackend;

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

  virtual ~PDBPhysicalAlgorithm() = default;

  PDBPhysicalAlgorithm(const std::string &firstTupleSet,
                       const std::string &finalTupleSet,
                       const pdb::Handle<PDBSourcePageSetSpec> &source,
                       const pdb::Handle<PDBSinkPageSetSpec> &sink,
                       const pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> &secondarySources)
      : firstTupleSet(firstTupleSet), finalTupleSet(finalTupleSet), source(source), sink(sink), secondarySources(secondarySources) {}

  /**
   * Sets up the whole algorithm
   */
  virtual bool setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) { throw std::runtime_error("Can not setup PDBPhysicalAlgorithm that is an abstract class"); };

  /**
   * Runs the algorithm
   */
  virtual bool run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) { throw std::runtime_error("Can not run PDBPhysicalAlgorithm that is an abstract class"); };

  /**
   * Returns the type of the algorithm we want to run
   */
  virtual PDBPhysicalAlgorithmType getAlgorithmType() { throw std::runtime_error("Can not get the type of the base class"); };

protected:

  /**
   * The source the algorithm should setup
   */
  pdb::Handle<PDBSourcePageSetSpec> source;

  /**
   * The sink the algorithm should setup
   */
  pdb::Handle<PDBSinkPageSetSpec> sink;

  /**
   * This is the tuple set of the atomic computation from which we are starting our pipeline
   */
  pdb::String firstTupleSet;

  /**
   * The is the tuple set of the atomic computation where we are ending our pipeline
   */
  pdb::String finalTupleSet;

  /**
   * List of secondary sources like hash sets for join etc.. null if there are no secondary sources
   */
  pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> secondarySources;

  // mark the tests that are testing this algorithm
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregation);
};

}

#endif //PDB_PDBPHYSICALALGORITHM_H
