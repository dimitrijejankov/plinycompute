#pragma once

#include <PipelineInterface.h>
#include <cstdio>
#include <list>
#include <PDBAnonymousPageSet.h>
#include <AggregationCombinerSink.h>

namespace pdb {

class AggregationPipeline : public PipelineInterface {
private:

  // the id of the worker this pipeline is running on
  size_t workerID;

  // this is the page set where we are going to be writing the output hash table
  pdb::PDBAnonymousPageSetPtr outputPageSet;

  // this is the page set where we are reading the hash sets to combine from
  pdb::PDBAbstractPageSetPtr inputPageSet;

  // the merger sink
  pdb::AggregationCombinerSinkBasePtr merger;

  // worker queue
  pdb::PDBWorkerQueuePtr workerQueue;

public:

  AggregationPipeline(size_t workerID,
                      PDBAnonymousPageSetPtr outputPageSet,
                      PDBAbstractPageSetPtr inputPageSet,
                      PDBWorkerQueuePtr workerQueue,
                      const pdb::ComputeSinkPtr &merger);

  void run() override;

};

}