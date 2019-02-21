//
// Created by dimitrije on 2/21/19.
//

#ifndef PDB_PDBABSTRACTPIPELINE_H
#define PDB_PDBABSTRACTPIPELINE_H

#include <AtomicComputation.h>

namespace pdb {

class PDBAbstractPipeline {

public:

  /**
   * Where the pipeline begins
   */
  AtomicComputationPtr sourceComputation;

  /**
   * Where the pipeline ends
   */
  AtomicComputationPtr targetComputation;

  /**
   * Returns the cost of running this pipeline
   * @return the cost
   */
  virtual size_t getCost() {
    throw std::runtime_error("");
  }

};

}
#endif //PDB_PDBABSTRACTPIPELINE_H
