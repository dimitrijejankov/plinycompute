//
// Created by dimitrije on 3/27/19.
//

#ifndef PDB_ABSTRACTPIPELINE_H
#define PDB_ABSTRACTPIPELINE_H

#include <memory>

namespace pdb {

class PipelineInterface {
public:

  virtual ~PipelineInterface() = default;

  /**
   * Runs the pipeline
   */
  virtual void run() = 0;
};

typedef std::shared_ptr<PipelineInterface> PipelinePtr;

}


#endif //PDB_ABSTRACTPIPELINE_H
