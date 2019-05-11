#pragma once

#include "PDBPhysicalAlgorithm.h"

// PRELOAD %PDBShuffleForJoinAlgorithm%

namespace pdb {

class PDBShuffleForJoinAlgorithm : public PDBPhysicalAlgorithm {
public:

  PDBShuffleForJoinAlgorithm() = default;

  PDBShuffleForJoinAlgorithm(const std::string &firstTupleSet,
                             const std::string &finalTupleSet,
                             const pdb::Handle<pdb::PDBSourcePageSetSpec> &source,
                             const pdb::Handle<pdb::PDBSinkPageSetSpec> &intermediate,
                             const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                             const pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &secondarySources);

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
  void cleanup() override {  };

 private:

  /**
   * The intermediate page set
   */
  pdb::Handle<PDBSinkPageSetSpec> intermediate;

  FRIEND_TEST(TestPhysicalOptimizer, TestJoin2);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin3);
};

}