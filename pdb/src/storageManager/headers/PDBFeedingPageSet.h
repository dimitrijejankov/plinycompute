//
// Created by dimitrije on 3/8/19.
//

#ifndef PDB_PDBFEEDINGPAGESET_H
#define PDB_PDBFEEDINGPAGESET_H

#include "PDBAbstractPageSet.h"
#include <map>
#include <condition_variable>
#include <PDBBufferManagerInterface.h>
#include <PDBFeedingPageSet.h>

namespace pdb {


// just make the ptr
class PDBFeedingPageSet;
using PDBFeedingPageSetPtr = std::shared_ptr<pdb::PDBFeedingPageSet>;

struct PDBFeedingPageInfo {

  PDBFeedingPageInfo(PDBPageHandle page, uint64_t numUsers, uint64_t timesServed);

  /**
   *
   */
  PDBPageHandle page = nullptr;

  /**
   *
   */
  uint64_t numUsers = 0;

  /**
   *
   */
  uint64_t timesServed = 0;
};

class PDBFeedingPageSet : public PDBAbstractPageSet {

public:

  /**
   *
   * @param numReaders
   * @param numFeeders
   */
  PDBFeedingPageSet(uint64_t numReaders, uint64_t numFeeders);

  /**
   * Returns the next page for are particular worker.
   * @param workerID - the id of the worker
   * @return the page
   */
  PDBPageHandle getNextPage(size_t workerID) override;

  /**
   * This method is not implemented in the feeding page set since it gets it's pages from the @see addPageSet.
   * If called this method will throw a runtime error
   * @return - throws exception
   */
  PDBPageHandle getNewPage() override;

  /**
   * Add the page to a page set.
   * @param page - the page we want to add
   */
  void feedPage(const PDBPageHandle &page);

  /**
   * Call when one of the feeders has finished feeding pages
   */
  void finishFeeding();

  /**
   * Return the number of pages in this page set
   * @return - the number of pages
   */
  size_t getNumPages() override;

private:

  /**
   * Keeps track of all anonymous pages so that we can quickly remove them
   */
  std::map<uint64_t, PDBFeedingPageInfo> pages;

  /**
   * Keeps track for each worker what was the last page it got.
   */
  std::vector<uint64_t> nextPageForWorker;

  /**
   * How many feeders have finished now
   */
  uint64_t numFinishedFeeders;

  /**
   *
   */
  uint64_t numFeeders;

  uint64_t numReaders;

  /**
   * This tells us what is the id of the next page id when feeding
   */
  uint64_t nextPage;

  /**
   * Mutex to sync the pages map
   */
  std::mutex m;

  /**
   * The condition variable to make the workers wait when needing pages
   */
  std::condition_variable cv{};
};



}

#endif //PDB_PDBFeedingPageSet_H
