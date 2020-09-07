#pragma once

// PRELOAD %TRALocalJoin%

#include <gtest/gtest_prod.h>
#include <PipelineInterface.h>
#include <processors/PreaggregationPageProcessor.h>
#include "PDBPhysicalAlgorithm.h"
#include "PDBPageSelfReceiver.h"
#include "Computation.h"
#include "PDBPageNetworkSender.h"

namespace pdb {

class TRALocalJoin : public PDBPhysicalAlgorithm {
 public:

  ENABLE_DEEP_COPY

  TRALocalJoin() = default;

  ~TRALocalJoin() override = default;

  TRALocalJoin(const std::string &lhsPageSet,
               const std::vector<int32_t>& lhs_indices,
               const std::string &rhsDb,
               const std::string &rhsSet,
               const std::vector<int32_t>& rhs_indices,
               const std::string &pageSet);

  /**
   *
   * @param job
   * @return
   */
  [[nodiscard]] PDBPhysicalAlgorithmStatePtr getInitialState(const pdb::Handle<pdb::ExJob> &job) const override;

  /**
   *
   * @return
   */
  [[nodiscard]] PDBPhysicalAlgorithmStagePtr getNextStage(const PDBPhysicalAlgorithmStatePtr &state) override;

  /**
   *
   * @return
   */
  [[nodiscard]] int32_t numStages() const override;

  /**
   * Returns BroadcastForJoin as the type
   * @return the type
   */
  PDBPhysicalAlgorithmType getAlgorithmType() override;

  /**
   * The pages of the aggregation always have as the root object pdb::map so it returns PDB_CATALOG_SET_MAP_CONTAINER
   * @return PDB_CATALOG_SET_MAP_CONTAINER
   */
  PDBCatalogSetContainerType getOutputContainerType() override;

 private:

  // source db
  pdb::String db;

  // source set
  pdb::String set;

  // the page
  pdb::String inputPageSet;

  // sink
  pdb::String sink;

  // indices
  pdb::Vector<int32_t> lhs_indices;

  int32_t currentStage = 0;
};

}