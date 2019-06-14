#pragma once

#include "PDBPhysicalAlgorithm.h"
#include <ComputePlan.h>
#include <ExJob.h>
#include <PDBStorageManagerBackend.h>
#include <PDBPageNetworkSender.h>
#include <BroadcastJoinProcessor.h>
#include <PDBPageSelfReceiver.h>
#include <GenericWork.h>
#include <memory>


// PRELOAD %PDBBroadcastForJoinAlgorithm%

namespace pdb {

class PDBBroadcastForJoinAlgorithm : public PDBPhysicalAlgorithm {
 public:

  PDBBroadcastForJoinAlgorithm() = default;

  PDBBroadcastForJoinAlgorithm(const std::string &firstTupleSet,
                               const std::string &finalTupleSet,
                               const pdb::Handle<pdb::PDBSourcePageSetSpec> &source,
                               const pdb::Handle<pdb::PDBSinkPageSetSpec> &intermediate,
                               const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                               const pdb::Handle<pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>>> &secondarySources,
                               const bool swapLHSandRHS);

  ENABLE_DEEP_COPY

  /**
   * Returns DistributedAggregation as the type
   * @return the type
   */
  PDBPhysicalAlgorithmType getAlgorithmType() override;

  /**
   * //TODO
   */
  bool setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
             Handle<pdb::ExJob> &job,
             const std::string &error) override;

  /**
   * //TODO
   */
  bool run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

  void cleanup() override;

 private:

  /**
   * The intermediate page set
   */
  pdb::Handle<PDBSinkPageSetSpec> intermediate;

  /**
   * This forwards the preaggregated pages to this node
   */
  pdb::PDBPageSelfReceiverPtr selfReceiver;

  /**
   * These senders forward pages that are for other nodes
   */
  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> senders;

  /**
   *
   */
  PDBLoggerPtr logger;

  /**
   * Vector of pipelines that will run this algorithm. The pipelines will be built when you call setup on this object.
   * This must be null when sending this object.
   */
  std::shared_ptr<std::vector<PipelinePtr>> prebroadcastjoinPipelines = nullptr;

  /**
   *
   */
  std::shared_ptr<std::vector<PipelinePtr>> broadcastjoinPipelines = nullptr;

  /**
   *
   */
  std::shared_ptr<std::vector<PDBPageQueuePtr>> pageQueues = nullptr;

  /**
   *
   */
  LogicalPlanPtr logicalPlan;

  // mark the tests that are testing this algorithm
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin1);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin3);
};

}