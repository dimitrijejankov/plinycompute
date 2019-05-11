#pragma once

#include "PDBPhysicalAlgorithm.h"

namespace pdb {

class PDBShuffleForJoinAlgorithm : public PDBPhysicalAlgorithm {
public:

  PDBShuffleForJoinAlgorithm(const std::string &firstTupleSet,
                             const std::string &finalTupleSet,
                             const pdb::Handle<pdb::PDBSourcePageSetSpec> &source,
                             const pdb::Handle<pdb::PDBSinkPageSetSpec> &intermediate,
                             const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                             const pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &secondarySources);

  /**
   * Returns ShuffleForJoinAlgorithm as the type
   * @return the type
   */
  PDBPhysicalAlgorithmType getAlgorithmType() override;

private:

  /**
   * The intermediate page set
   */
  pdb::Handle<PDBSinkPageSetSpec> intermediate;

  FRIEND_TEST(TestPhysicalOptimizer, TestJoin2);
  FRIEND_TEST(TestPhysicalOptimizer, TestJoin3);
};

}