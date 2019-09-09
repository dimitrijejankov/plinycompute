//
// Created by dimitrije on 3/20/19.
//

#ifndef PDB_PDBAGGREGATIONPIPEALGORITHM_H
#define PDB_PDBAGGREGATIONPIPEALGORITHM_H

// PRELOAD %PDBAggregationPipeAlgorithm%

#include <gtest/gtest_prod.h>
#include <PipelineInterface.h>
#include <processors/PreaggregationPageProcessor.h>
#include "PDBPhysicalAlgorithm.h"
#include "PDBPageSelfReceiver.h"
#include "Computation.h"
#include "PDBPageNetworkSender.h"

namespace pdb {

class PDBAggregationPipeAlgorithm : public PDBPhysicalAlgorithm {
public:

  ENABLE_DEEP_COPY

  PDBAggregationPipeAlgorithm() = default;

  ~PDBAggregationPipeAlgorithm() override = default;

  PDBAggregationPipeAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                              const AtomicComputationPtr &finalAtomicComputation,
                              const Handle<PDBSinkPageSetSpec> &hashedToSend,
                              const Handle<PDBSourcePageSetSpec> &hashedToRecv,
                              const Handle<PDBSinkPageSetSpec> &sink,
                              const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                              const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize);

  bool setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) override;

  bool run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

  void cleanup() override;

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

  /**
   * The sink tuple set where we are putting stuff
   */
  pdb::Handle<PDBSinkPageSetSpec> hashedToSend;

  /**
   * The sink type the algorithm should setup
   */
  pdb::Handle<PDBSourcePageSetSpec> hashedToRecv;

  /**
   * This forwards the preaggregated pages to this node
   */
  pdb::PDBPageSelfReceiverPtr selfReceiver;

  /**
   * These senders forward pages that are for other nodes
   */
  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> senders;

  /**
   * The logger for this algorithm, initialized on setup
   */
  PDBLoggerPtr logger;

  /**
   * Vector of pipelines that will run this algorithm. The pipelines will be built when you call setup on this object.
   * This must be null when sending this object.
   */
  std::shared_ptr<std::vector<PipelinePtr>> preaggregationPipelines = nullptr;

  /**
   *
   */
  std::shared_ptr<std::vector<PipelinePtr>> aggregationPipelines = nullptr;

  /**
   *
   */
  std::shared_ptr<std::vector<PDBPageQueuePtr>> pageQueues = nullptr;


  // mark the tests that are testing this algorithm
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregation);
  FRIEND_TEST(TestPhysicalOptimizer, TestMultiSink);
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregationAfterTwoWayJoin);
};

}

#endif //PDB_PDBAGGREGATIONPIPEALGORITHM_H
