#include <utility>

//
// Created by yuxin on 3/29/19.
//

#ifndef PDB_BROADCAST_JOIN_PAGEPROCESSOR_H
#define PDB_BROADCAST_JOIN_PAGEPROCESSOR_H

#include <PageProcessor.h>
#include <PDBPageHandle.h>
#include <PDBBufferManagerInterface.h>
#include <JoinMap.h>

namespace pdb {

/**
 * This is the processor for the pages that contain the result of the preaggregation
 *
 */
class BroadcastJoinProcessor : public PageProcessor {
 public:

  BroadcastJoinProcessor() = default;

  ~BroadcastJoinProcessor() override = default;

  BroadcastJoinProcessor(size_t numNodes,
                         size_t numProcessingThreads,
                         vector<PDBPageQueuePtr> pageQueues,
                         PDBBufferManagerInterfacePtr bufferManager) : numNodes(numNodes),
                                                                       numProcessingThreads(numProcessingThreads),
                                                                       pageQueues(std::move(pageQueues)),
                                                                       bufferManager(std::move(bufferManager)) {}

  bool process(const MemoryHolderPtr &memory) override {
    // if we do not have a sink just finish
    if (memory->outputSink == nullptr) {
      return true;
    }
    // put all the objects into Vector<JoinMap>
    pdb::Handle<pdb::Vector<pdb::Handle<pdb::JoinMap<pdb::Object>>>>
        allMaps = unsafeCast<pdb::Vector<pdb::Handle<pdb::JoinMap<pdb::Object>>>>(memory->outputSink);

    auto record = getRecord(allMaps);

    memory->pageHandle->freezeSize(record->numBytes());

    //memory->pageHandle->unpin();

    for (auto node = 0; node < numNodes; ++node) {
      pageQueues[node]->enqueue(memory->pageHandle);
    }

    memory->pageHandle->unpin();
    return false;
  }

 private:

  /**
   * The number of nodes we have
   */
  size_t numNodes = 0;

  /**
   * The number of processing threads per node
   */
  size_t numProcessingThreads = 0;

  /**
   * Where we put the pages
   */
  std::vector<PDBPageQueuePtr> pageQueues;

  /**
   * The buffer manager
   */
  PDBBufferManagerInterfacePtr bufferManager;

};

}

#endif //PDB_BROADCAST_JOIN_PAGEPROCESSOR_H
