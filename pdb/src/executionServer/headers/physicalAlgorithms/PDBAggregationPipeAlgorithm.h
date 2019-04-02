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

namespace pdb {

class PDBAggregationPipeAlgorithm : public PDBPhysicalAlgorithm {
public:

  ENABLE_DEEP_COPY

  PDBAggregationPipeAlgorithm() = default;

  ~PDBAggregationPipeAlgorithm() override = default;

  PDBAggregationPipeAlgorithm(const std::string &firstTupleSet,
                              const std::string &finalTupleSet,
                              const Handle<PDBSourcePageSetSpec> &source,
                              const Handle<PDBSinkPageSetSpec> &hashedToSend,
                              const Handle<PDBSourcePageSetSpec> &hashedToRecv,
                              const Handle<PDBSinkPageSetSpec> &sink,
                              const Handle<Vector<PDBSourcePageSetSpec>> &secondarySources);

  bool setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) override;

  bool run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

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
   * Vector of pipelines that will run this algorithm. The pipelines will be built when you call setup on this object.
   * This must be null when sending this object.
   */
  std::shared_ptr<std::vector<PipelinePtr>> pipelines = nullptr;

  std::shared_ptr<std::vector<preaggPageQueuePtr>> pageQueues = nullptr;

  // mark the tests that are testing this algorithm
  FRIEND_TEST(TestPhysicalOptimizer, TestAggregation);
};

}

#endif //PDB_PDBAGGREGATIONPIPEALGORITHM_H
