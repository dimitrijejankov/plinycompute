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
#include <PDBSetObject.h>
#include <PDBCatalogSet.h>
#include <LogicalPlan.h>
#include <SourceSetArg.h>
#include <PDBVector.h>
#include <JoinArguments.h>
#include <PDBSourceSpec.h>
#include <gtest/gtest_prod.h>
#include <physicalOptimizer/PDBPrimarySource.h>

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

  PDBPhysicalAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                       const AtomicComputationPtr &finalAtomicComputation,
                       const pdb::Handle<PDBSinkPageSetSpec> &sink,
                       const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                       const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize);

  /**
   * Sets up the whole algorithm
   */
  virtual bool setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) { throw std::runtime_error("Can not setup PDBPhysicalAlgorithm that is an abstract class"); };

  /**
   * Runs the algorithm
   */
  virtual bool run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) { throw std::runtime_error("Can not run PDBPhysicalAlgorithm that is an abstract class"); };

  /**
   * Cleans the algorithm after setup and/or run. This has to be called after the usage!
   */
  virtual void cleanup()  { throw std::runtime_error("Can not clean PDBPhysicalAlgorithm that is an abstract class"); };

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

  /**
   * Returns the source page set we are scanning.
   * @param storage - a ptr to the storage manager backend so we can grab the page set
   * @return - the page set
   */
  PDBAbstractPageSetPtr getSourcePageSet(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, size_t idx);

  /**
   * Return the info that is going to be provided to the pipeline about the main source set we are scanning
   * @return an instance of SourceSetArgPtr
   */
  pdb::SourceSetArgPtr getSourceSetArg(std::shared_ptr<pdb::PDBCatalogClient> &catalogClient, size_t idx);

  /**
   * Returns the additional sources as join arguments, if we can not find a page set that is specified in the additional sources
   * this method will return null
   * @param storage - Storage manager backend
   * @return the arguments if we can create them, null_ptr otherwise
   */
  std::shared_ptr<JoinArguments> getJoinArguments(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage);

  /**
   *
   */
  pdb::Vector<PDBSourceSpec> sources;

  /**
   * The is the tuple set of the atomic computation where we are ending our pipeline
   */
  pdb::String finalTupleSet;

  /**
   * List of secondary sources like hash sets for join etc.. null if there are no secondary sources
   */
  pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> secondarySources;

  /**
   * The sink page set the algorithm should setup
   */
  pdb::Handle<PDBSinkPageSetSpec> sink;

  /**
   * The sets we want to materialize the result of this aggregation to
   */
  pdb::Handle<pdb::Vector<PDBSetObject>> setsToMaterialize;

  /**
   * The logical plan
   */
  pdb::LogicalPlanPtr logicalPlan;

  // mark the tests that are testing this algorithm
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregation);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin1);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin2);
  FRIEND_TEST(TestPhysicalOptimizer, TestMultiSink);
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregationAfterTwoWayJoin);
};

}

#endif //PDB_PDBPHYSICALALGORITHM_H
