//
// Created by dimitrije on 3/20/19.
//

#ifndef PDB_PDBAGGREGATIONPIPEALGORITHM_H
#define PDB_PDBAGGREGATIONPIPEALGORITHM_H

// PRELOAD %PDBAggregationPipeAlgorithm%

#include <gtest/gtest_prod.h>
#include "PDBPhysicalAlgorithm.h"

namespace pdb {

class PDBAggregationPipeAlgorithm : public PDBPhysicalAlgorithm {
public:

  PDBAggregationPipeAlgorithm() = default;

  ~PDBAggregationPipeAlgorithm() override = default;

  PDBAggregationPipeAlgorithm(const std::string &firstTupleSet,
                              const std::string &finalTupleSet,
                              const Handle<PDBSourcePageSetSpec> &source,
                              const Handle<PDBSinkPageSetSpec> &hashedToSend,
                              const Handle<PDBSourcePageSetSpec> &hashedToRecv,
                              const Handle<PDBSinkPageSetSpec> &sink,
                              const Handle<Vector<PDBSourcePageSetSpec>> &secondarySources);

private:

  /**
   * The sink tuple set where we are putting stuff
   */
  pdb::Handle<PDBSinkPageSetSpec> hashedToSend;

  /**
   * The sink type the algorithm should setup
   */
  pdb::Handle<PDBSourcePageSetSpec> hashedToRecv;

  // mark the tests that are testing this algorithm
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregation);
};

}

#endif //PDB_PDBAGGREGATIONPIPEALGORITHM_H
