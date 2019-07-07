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

enum class PDBFeedingPageSetUsagePolicy {

  REMOVE_AFTER_USED,
  KEEP_AFTER_USED
};

class PDBFeedingPageSet;
using PDBFeedingPageSetPtr = std::shared_ptr<pdb::PDBFeedingPageSet>;

/**
 * Internal structure for the @see PDBFeedingPageSet
 */
struct PDBFeedingPageInfo {

  PDBFeedingPageInfo(PDBPageHandle page, uint64_t numUsers, uint64_t timesServed);

  /**
   * The page handle of the page we stored in the @see PDBFeedingPageSet
   */
  PDBPageHandle page = nullptr;

  /**
   * The number of current users of that page
   */
  uint64_t numUsers = 0;

  /**
   * How many times have we served this page to a reader
   */
  uint64_t timesServed = 0;
};

/**
 * This page set basically gets pages feed into it so multiple threads can read them.
 * The constructor takes in how many threads are going to feed (put) pages into it, and how many threads are going to
 * read pages from it. Each reading thread will receive each page put into the page set. It is important to note that,
 * the if there are no pages and not all feeders have finished feeding pages the method getNextPage will block.
 */
class PDBFeedingPageSet : public PDBAbstractPageSet {

 public:

  /**
   * Initializes the feeding page set
   * @param numReaders - how many threads are going to read pages from it
   * @param numFeeders - how many threads are going to feed pages into it
   */
  PDBFeedingPageSet(uint64_t numReaders, uint64_t numFeeders);

  /**
   * Specifies what to do we do with the pages once all the readers are done accessing?
   * @param policy
   */
  void setUsagePolicy(PDBFeedingPageSetUsagePolicy policy);

  /**
   * Returns the next page for are particular worker. This is a blocking method, and it will stall until we get a page,
   * or all the feeders are finished feeding.
   * @param workerID - the id of the worker
   * @return the page or null if we are done reading pages,
   */
  PDBPageHandle getNextPage(size_t workerID) override;

  /**
   * This method is not implemented in the feeding page set since it gets it's pages from the @see feedPage.
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

  /**
   * Resets the page set so it can be reused
   */
  void resetPageSet() override;

 private:

  /**
   * What do we do with the pages once all the readers are done accessing?
   */
  PDBFeedingPageSetUsagePolicy usagePolicy = PDBFeedingPageSetUsagePolicy::REMOVE_AFTER_USED;

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
   * How many threads are feeding pages into this page set
   */
  uint64_t numFeeders;

  /**
   * The number of threads reading page from this page set
   */
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
