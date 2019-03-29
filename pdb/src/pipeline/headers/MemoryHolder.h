//
// Created by dimitrije on 3/27/19.
//

#ifndef PDB_MEMORYHOLDER_H
#define PDB_MEMORYHOLDER_H

#include <Object.h>
#include <PDBPageHandle.h>
#include <Handle.h>

namespace pdb {

// this is used to buffer unwritten pages
struct MemoryHolder {
  // the output vector that this guy stores
  Handle<Object> outputSink;
  // page handle
  PDBPageHandle pageHandle;
  // the iteration where he was last written...
  // we use this because we cannot delete
  int iteration;
  void setIteration(int iterationIn);
  explicit MemoryHolder(const PDBPageHandle &pageHandle);
};

typedef std::shared_ptr<MemoryHolder> MemoryHolderPtr;

}

#endif //PDB_MEMORYHOLDER_H
