//
// Created by dimitrije on 4/11/19.
//

#ifndef PDB_JOINBROADCASTPIPELINE_H
#define PDB_JOINBROADCASTPIPELINE_H

#include <PipelineInterface.h>
#include <PDBAnonymousPageSet.h>
#include <ComputeSink.h>

namespace pdb {

class JoinBroadcastPipeline : public PipelineInterface {
 private:

  // the id of the worker this pipeline is running on
  size_t workerID;

  // the id of the node the worker is running on
  size_t nodeID;

  // this is the page set where we are going to be writing the output hash table
  pdb::PDBAnonymousPageSetPtr outputPageSet;

  // this is the page set where we are reading the hash sets to combine from
  pdb::PDBAbstractPageSetPtr inputPageSet;

  // the merger sink
  pdb::ComputeSinkPtr merger;

 public:

  JoinBroadcastPipeline(size_t workerID,
                        PDBAnonymousPageSetPtr outputPageSet,
                        PDBAbstractPageSetPtr inputPageSet,
                        ComputeSinkPtr merger);

  void run() override;

};

}

#endif //PDB_JOINSHUFFLEPIPELINE_H
