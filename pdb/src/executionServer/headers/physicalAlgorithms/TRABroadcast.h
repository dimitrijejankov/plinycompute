#pragma once

// PRELOAD %TRABroadcast%

#include <gtest/gtest_prod.h>
#include <PipelineInterface.h>
#include <processors/PreaggregationPageProcessor.h>
#include "PDBPhysicalAlgorithm.h"
#include "PDBPageSelfReceiver.h"
#include "Computation.h"
#include "PDBPageNetworkSender.h"

namespace pdb {

class TRABroadcast : public PDBPhysicalAlgorithm {
 public:

  ENABLE_DEEP_COPY

  TRABroadcast() = default;

  ~TRABroadcast() override = default;

  TRABroadcast(const std::string &inputPageSet, const std::string &sink);

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

  // the page
  pdb::String inputPageSet;

  // sink
  pdb::String sink;

  int32_t currentStage = 0;
};

}