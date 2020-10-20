#pragma once

// PRELOAD %MM3D%

#include <gtest/gtest_prod.h>
#include <PipelineInterface.h>
#include <processors/PreaggregationPageProcessor.h>
#include "PDBPhysicalAlgorithm.h"
#include "PDBPageSelfReceiver.h"
#include "Computation.h"
#include "PDBPageNetworkSender.h"

namespace pdb {

class MM3D : public PDBPhysicalAlgorithm {
 public:

  ENABLE_DEEP_COPY

  MM3D() = default;

  ~MM3D() override = default;

  explicit MM3D(int32_t n, int32_t num_nodes, int32_t num_threads);

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

  // the number of nodes
  int32_t num_nodes;

  // the number of threads per node
  int32_t num_threads;

  // the number of compute sites in the cluster
  int32_t n;
};

}