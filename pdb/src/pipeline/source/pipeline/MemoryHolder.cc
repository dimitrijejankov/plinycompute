//
// Created by dimitrije on 3/27/19.
//

#include "MemoryHolder.h"

void pdb::MemoryHolder::setIteration(int iterationIn) {

  if (outputSink != nullptr) {
    getRecord(outputSink);
  }

  iteration = iterationIn;
}

pdb::MemoryHolder::MemoryHolder(const pdb::PDBPageHandle &pageHandle) {
  // set the page handle
  this->pageHandle = pageHandle;

  // make the allocation block
  makeObjectAllocatorBlock(this->pageHandle->getBytes(), this->pageHandle->getSize(), true);
  outputSink = nullptr;
}