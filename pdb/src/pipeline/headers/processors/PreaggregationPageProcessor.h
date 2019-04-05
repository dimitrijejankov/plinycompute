#include <utility>

//
// Created by dimitrije on 3/29/19.
//

#ifndef PDB_PRAGGREGATIONPAGEPROCESSOR_H
#define PDB_PRAGGREGATIONPAGEPROCESSOR_H

#include <PageProcessor.h>
#include <PDBPageHandle.h>
#include <PDBBufferManagerInterface.h>

namespace pdb {

/**
 * This is the processor for the pages that contain the result of the preaggregation
 * 
 */
class PreaggregationPageProcessor : public PageProcessor  {
public:

  PreaggregationPageProcessor() = default;

  ~PreaggregationPageProcessor() override = default;

  PreaggregationPageProcessor(size_t numNodes,
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
    pdb::Handle<pdb::Vector<pdb::Handle<pdb::Map<pdb::Nothing>>>> allMaps = unsafeCast<pdb::Vector<pdb::Handle<pdb::Map<pdb::Nothing>>>>(memory->outputSink);

    for(auto node = 0; node < numNodes; ++node) {

      // get the page
      auto page = bufferManager->getPage();

      // set it as the current allocation block
      const pdb::UseTemporaryAllocationBlock tempBlock{page->getBytes(), page->getSize()};

      // make an object to hold
      pdb::Handle<pdb::Vector<pdb::Handle<pdb::Map<pdb::Nothing>>>> maps = pdb::makeObject<pdb::Vector<pdb::Handle<pdb::Map<pdb::Nothing>>>>();

      // copy all the maps  that we need to
      for(int t = 0; t < numProcessingThreads; ++t) {

        // copy the map
        maps->push_back((*allMaps)[node * numProcessingThreads + t]);
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
