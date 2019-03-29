//
// Created by dimitrije on 3/27/19.
//

#ifndef PDB_AGGREGATIONPIPELINE_H
#define PDB_AGGREGATIONPIPELINE_H

#include <PipelineInterface.h>
#include <cstdio>
#include <PDBAnonymousPageSet.h>
#include <ComputeSink.h>

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
  pdb::ComputeSinkPtr merger;

public:

  AggregationPipeline(size_t workerID,
                      const PDBAnonymousPageSetPtr &outputPageSet,
                      const PDBAbstractPageSetPtr &inputPageSet,
                      const ComputeSinkPtr &merger);

  void run() override;

};

}


#endif //PDB_AGGREGATIONPIPELINE_H
