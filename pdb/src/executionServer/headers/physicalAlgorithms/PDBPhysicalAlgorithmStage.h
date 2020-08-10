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
#include <PDBStorageManagerFrontend.h>

namespace pdb {

// predefine this so avoid recursive definition
class ExJob;

/**
 * Abstracts all the functionality of a physical algorithm stage
 */
class PDBPhysicalAlgorithmStage {
public:

  PDBPhysicalAlgorithmStage(const PDBSinkPageSetSpec &sink,
                            const Vector<PDBSourceSpec> &sources,
                            const String &finalTupleSet,
                            const Vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                            const Vector<PDBSetObject> &setsToMaterialize)
      : sink(sink),
        sources(sources),
        finalTupleSet(finalTupleSet),
        secondarySources(secondarySources),
        setsToMaterialize(setsToMaterialize) {}

  /**
   * Sets up the stage
   */
  virtual bool setup(const Handle<pdb::ExJob> &job, const PDBPhysicalAlgorithmStatePtr &state,
                     const std::shared_ptr<pdb::PDBStorageManagerFrontend> &storage, const std::string &error) {
    throw std::runtime_error("Can not setup PDBPhysicalAlgorithmStage that is an abstract class");
  };

  /**
   * Runs the stage
   */
  virtual bool run(const Handle<pdb::ExJob> &request, const PDBPhysicalAlgorithmStatePtr &state,
                   const std::shared_ptr<pdb::PDBStorageManagerFrontend> &storage, const std::string &error) {
    throw std::runtime_error("Can not run PDBPhysicalAlgorithmStage that is an abstract class");
  };

  /**
   * Cleans the stage after setup and/or run. This has to be called after the usage!
   */
  virtual void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerFrontend> &storage)  {
    throw std::runtime_error("Can not clean PDBPhysicalAlgorithmStage that is an abstract class");
  };

protected:

  /**
   * Returns a key extractor for the computation that is generating the final tuple set there
   * @param finalTupleSet - the final tuple set
   * @param plan - the plan we use for lookup
   * @return the key extractor
   */
  static PDBKeyExtractorPtr getKeyExtractor(const std::string &finalTupleSet, ComputePlan &plan);

  /**
   * Returns the source page set we are scanning.
   * @param storage - a ptr to the storage manager backend so we can grab the page set
   * @return - the page set
   */
  PDBAbstractPageSetPtr getSourcePageSet(const std::shared_ptr<pdb::PDBStorageManagerFrontend> &storage, size_t idx);

  /**
   * Return the info that is going to be provided to the pipeline about the main source set we are scanning
   * @return an instance of SourceSetArgPtr
   */
  pdb::SourceSetArgPtr getSourceSetArg(const std::shared_ptr<pdb::PDBCatalogClient> &catalogClient, size_t idx);

  /**
   * Returns the additional sources as join arguments, if we can not find a page set that is specified in the additional sources
   * this method will return null
   * @param storage - Storage manager backend
   * @return the arguments if we can create them, null_ptr otherwise
   */
  std::shared_ptr<JoinArguments> getJoinArguments(const std::shared_ptr<pdb::PDBStorageManagerFrontend> &storage);

  // The sink page set the algorithm should setup
  const PDBSinkPageSetSpec &sink;

  // The primary sources of the pipeline
  const pdb::Vector<PDBSourceSpec> &sources;

  // The is the tuple set of the atomic computation where we are ending our pipeline
  const pdb::String &finalTupleSet;

  // List of secondary sources like hash sets for join etc.. null if there are no secondary sources
  const pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources;

  // The sets we want to materialize the result of this aggregation to
  const pdb::Vector<PDBSetObject> &setsToMaterialize;

};

// the state ptr
using PDBPhysicalAlgorithmStagePtr = std::shared_ptr<PDBPhysicalAlgorithmStage>;

}