#pragma once

#include "PDBPhysicalAlgorithm.h"
#include "PageProcessor.h"
#include "PDBPageNetworkSender.h"
#include "PDBPageSelfReceiver.h"
#include "PipelineInterface.h"
#include "Computation.h"

// PRELOAD %PDBShuffleForJoinAlgorithm%

namespace pdb {

class PDBShuffleForJoinAlgorithm : public PDBPhysicalAlgorithm {
public:

  PDBShuffleForJoinAlgorithm() = default;

  PDBShuffleForJoinAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                             const AtomicComputationPtr &finalAtomicComputation,
                             const pdb::Handle<pdb::PDBSinkPageSetSpec> &intermediate,
                             const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                             const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                             const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize);

  ENABLE_DEEP_COPY

  /**
   * Returns ShuffleForJoinAlgorithm as the type
   * @return the type
   */
  PDBPhysicalAlgorithmType getAlgorithmType() override;

  /**
   * //TODO
   */
  bool setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) override;

  /**
   * //TODO
   */
  bool run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

  /**
   * //TODO
   */
  void cleanup() override;

 private:

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
  std::shared_ptr<std::vector<PipelinePtr>> joinShufflePipelines = nullptr;

  /**
   *
   */
  std::shared_ptr<std::vector<PDBPageQueuePtr>> pageQueues = nullptr;

  /**
   *
   */
  PDBLoggerPtr logger;

  /**
   * The intermediate page set
   */
  pdb::Handle<PDBSinkPageSetSpec> intermediate;

  FRIEND_TEST(TestPhysicalOptimizer, TestJoin2);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin3);
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregationAfterTwoWayJoin);
};

}