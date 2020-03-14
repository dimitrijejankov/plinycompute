#include <utility>

#pragma once

#include <Object.h>
#include <PDBString.h>
#include <PDBSourcePageSetSpec.h>
#include <PDBSinkPageSetSpec.h>
#include <PDBSetObject.h>
#include <PDBCatalogSet.h>
#include <LogicalPlan.h>
#include <SourceSetArg.h>
#include <PDBVector.h>
#include <JoinArguments.h>
#include <PDBSourceSpec.h>
#include <gtest/gtest_prod.h>
#include <physicalOptimizer/PDBPrimarySource.h>
#include <PDBPhysicalAlgorithmState.h>
#include "PDBPhysicalAlgorithmStage.h"

namespace pdb {

// predefine this so avoid recursive definition
class ExJob;
class PDBStorageManagerBackend;

enum PDBPhysicalAlgorithmType {

  ShuffleForJoin,
  BroadcastForJoin,
  DistributedAggregation,
  StraightPipe,
  JoinAggregation,
  EightWayJoin
};

// PRELOAD %PDBPhysicalAlgorithm%


class PDBPhysicalAlgorithm : public Object {
public:

  ENABLE_DEEP_COPY

  PDBPhysicalAlgorithm() = default;

  virtual ~PDBPhysicalAlgorithm() = default;

  PDBPhysicalAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                       const AtomicComputationPtr &finalAtomicComputation,
                       const pdb::Handle<PDBSinkPageSetSpec> &sink,
                       const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                       const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize);



  /**
   * Returns the initial state of the algorithm
   * @return the initial state
   */
  [[nodiscard]] virtual PDBPhysicalAlgorithmStatePtr getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
    throw std::runtime_error("Can not get the type of the base class");
  };

  /**
   * Returns all the stages of this algorithm
   * @return
   */
  [[nodiscard]] virtual std::vector<PDBPhysicalAlgorithmStagePtr> getStages() const {
    throw std::runtime_error("Can not get the type of the base class");
  };


  /**
   * Returns the number of stages this algorithm has
   * @return
   */
  [[nodiscard]] virtual int32_t numStages() const {
    throw std::runtime_error("Can not get the type of the base class");
  };

  /**
   * Returns the type of the algorithm we want to run
   */
  virtual PDBPhysicalAlgorithmType getAlgorithmType() { throw std::runtime_error("Can not get the type of the base class"); };

  /**
   * Returns the all the that are about to be materialized by the algorithm
   * @return the vector of @see PDBSetObject
   */
  const pdb::Handle<pdb::Vector<PDBSetObject>> &getSetsToMaterialize() { return setsToMaterialize; }

  /**
   * Returns the set this algorithm is going to scan
   * @return source set as @see PDBSetObject
   */
  std::vector<std::pair<std::string, std::string>> getSetsToScan() {

    // figure out the sets
    std::vector<std::pair<std::string, std::string>> tmp;
    for(int i = 0; i < sources.size(); ++i) {

      // if we have a set store it
      if(sources[i].sourceSet != nullptr) {
        tmp.emplace_back(std::make_pair<std::string, std::string>(sources[i].sourceSet->database, sources[i].sourceSet->set));
      }
    }

    // move the vector
    return std::move(tmp);
  }

  /**
   * Returns the type of the container that the materialized result will have
   */
  virtual pdb::PDBCatalogSetContainerType getOutputContainerType() { return PDB_CATALOG_SET_NO_CONTAINER; };

protected:


  // The sink page set the algorithm should setup
  pdb::Handle<PDBSinkPageSetSpec> sink;

  // The primary sources of the pipeline
  pdb::Vector<PDBSourceSpec> sources;

  // List of secondary sources like hash sets for join etc.. null if there are no secondary sources
  pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> secondarySources;

  // The is the tuple set of the atomic computation where we are ending our pipeline
  pdb::String finalTupleSet;

  // The sets we want to materialize the result of this aggregation to
  pdb::Handle<pdb::Vector<PDBSetObject>> setsToMaterialize;

  // mark the tests that are testing this algorithm
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregation);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin1);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin2);
  FRIEND_TEST(TestPhysicalOptimizer, TestMultiSink);
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregationAfterTwoWayJoin);
};

}