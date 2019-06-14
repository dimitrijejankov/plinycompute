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
                               const pdb::Handle<PDBSourcePageSetSpec> &source,
                               const pdb::Handle<PDBSinkPageSetSpec> &intermediate,
                               const pdb::Handle<PDBSinkPageSetSpec> &sink,
                               const pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &secondarySources,
                               bool swapLHSandRHS);

  ENABLE_DEEP_COPY

  /**
   * Returns DistributedAggregation as the type
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

  void cleanup() override {  };

private:

  /**
   * The intermediate page set
   */
  pdb::Handle<PDBSinkPageSetSpec> intermediate;

  // mark the tests that are testing this algorithm
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin1);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin3);
};

}