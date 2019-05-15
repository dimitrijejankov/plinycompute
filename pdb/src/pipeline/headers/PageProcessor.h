//
// Created by dimitrije on 3/29/19.
//

#ifndef PDB_PAGEPROCESSOR_H
#define PDB_PAGEPROCESSOR_H

#include "MemoryHolder.h"
#include "ComputeInfo.h"
#include <concurrent_queue.h>

namespace pdb {

using PDBPageQueuePtr = shared_ptr<concurent_queue<PDBPageHandle>>;
using PDBPageQueue = concurent_queue<PDBPageHandle>;

class PageProcessor;
using PageProcessorPtr = std::shared_ptr<PageProcessor>;

/**
 * This class processes the page that it gets from the pipeline. It can be used to do anything with the page, as long as it
 * does not modify the page or the sink. There is not guarantee that the outputSink is created. it could be that the page
 * only contains intermediate data.
 */
class PageProcessor : public ComputeInfo {
public:

  ~PageProcessor() override = default;

  /**
   * This method does the processing of the page
   * @param memory - the memory with the page and possibly the output sink, the output sink can be null
   * @return - true if we want to keep the page, false otherwise
   */
  virtual bool process(const MemoryHolderPtr &memory) = 0;

};

}
#endif //PDB_PAGEPROCESSOR_H
