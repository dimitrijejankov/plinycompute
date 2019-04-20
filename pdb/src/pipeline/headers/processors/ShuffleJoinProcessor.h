#include <utility>

//
// Created by dimitrije on 3/29/19.
//

#ifndef PDB_SHUFFLE_JOIN_PAGEPROCESSOR_H
#define PDB_SHUFFLE_JOIN_PAGEPROCESSOR_H

#include <PageProcessor.h>
#include <PDBPageHandle.h>
#include <PDBBufferManagerInterface.h>
#include <JoinMap.h>

namespace pdb {

/**
 * This is the processor for the pages that contain the result of the preaggregation
 * 
 */
template<typename RecordType>
class ShuffleJoinProcessor : public PageProcessor  {
public:

  ShuffleJoinProcessor() = default;

  ~ShuffleJoinProcessor() override = default;

  ShuffleJoinProcessor(size_t numNodes,
                       size_t numProcessingThreads,
                       vector<PDBPageQueuePtr> pageQueues,
                       PDBBufferManagerInterfacePtr bufferManager) : numNodes(numNodes),
                                                                     numProcessingThreads(numProcessingThreads),
                                                                     pageQueues(std::move(pageQueues)),
                                                                     bufferManager(std::move(bufferManager)) {}

  bool process(const MemoryHolderPtr &memory) override {

    // if we do not have a sink just finish
    if(memory->outputSink == nullptr) {
      return true;
    }

    // cast the thing to the maps of maps
    pdb::Handle<pdb::Vector<pdb::Handle<pdb::JoinMap<RecordType>>>> allMaps = unsafeCast<pdb::Vector<pdb::Handle<pdb::JoinMap<RecordType>>>>(memory->outputSink);

    for(auto node = 0; node < numNodes; ++node) {

      // get the page
      auto page = bufferManager->getPage();

      // set it as the current allocation block
      const pdb::UseTemporaryAllocationBlock tempBlock{page->getBytes(), page->getSize()};

      // make an object to hold
      pdb::Handle<pdb::Vector<pdb::Handle<pdb::JoinMap<RecordType>>>> maps = pdb::makeObject<pdb::Vector<pdb::Handle<pdb::JoinMap<RecordType>>>>();

      // copy all the maps  that we need to
      for(int t = 0; t < numProcessingThreads; ++t) {

        // deep copy the map
        pdb::Handle<pdb::JoinMap<RecordType>> copy = pdb::deepCopyJoinMap((*allMaps)[node * numProcessingThreads + t]);

        // copy the map
        maps->push_back(copy);
      }

      // get the record (this is important since it makes it the root object of the block)
      auto record = getRecord(maps);

      // freeze the page
      page->freezeSize(record->numBytes());

      // unpin the page
      page->unpin();

      // add the page to the page queue
      pageQueues[node]->enqueue(page);
    }

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


#endif //PDB_PRAGGREGATIONPAGEPROCESSOR_H
