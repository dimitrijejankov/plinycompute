//
// Created by dimitrije on 3/8/19.
//

#ifndef PDB_PDBANONYMOUSPAGESET_H
#define PDB_PDBANONYMOUSPAGESET_H

#include "PDBAbstractPageSet.h"
#include <map>
#include <PDBBufferManagerInterface.h>
#include <PDBAnonymousPageSet.h>

namespace pdb {

/**
 * This describes the access pattern of the page set
 */
enum class PDBAnonymousPageSetAccessPattern {

  SEQUENTIAL, // gives pages in order so basically each page can only be assigned to one worker
  CONCURRENT // pages assigned to multiple workers at the same time
};

// just make the ptr
class PDBAnonymousPageSet;
using PDBAnonymousPageSetPtr = std::shared_ptr<pdb::PDBAnonymousPageSet>;

class PDBAnonymousPageSet : public PDBAbstractPageSet {

public:

  PDBAnonymousPageSet() = default;

  explicit PDBAnonymousPageSet(const PDBBufferManagerInterfacePtr &bufferManager);

  /**
   * Returns the next page for are particular worker. This method is supposed to be used after all the pages have been
   * added that need to be added
   * @param workerID - the id of the worker
   * @return the page
   */
  PDBPageHandle getNextPage(size_t workerID) override;

  /**
   * Returns the new page for this page set. It is an anonymous page.
   * @return
   */
  PDBPageHandle getNewPage() override;

  /**
   * Return the number of pages in this page set
   * @return - the number of pages
   */
  size_t getNumPages() override;

  /**
   * Remove the page from this page. The page has to be in this page set, otherwise the behavior is not defined
   * @param pageHandle - the page handle we want to remove
   */
  void removePage(PDBPageHandle pageHandle);

  /**
   * Resets the page set so it can be reused
   */
  void resetPageSet() override;

  /**
   * Sets the access order for the page set
   * @param pattern - the pattern of access
   */
  void setAccessOrder(PDBAnonymousPageSetAccessPattern pattern);

 private:

  /**
   * Defines the iterator for the pages
   */
  using pageIterator = std::map<uint64_t, PDBPageHandle>::iterator;

  /**
   * Keeps track of all anonymous pages so that we can quickly remove them
   */
  std::map<uint64_t, PDBPageHandle> pages;

  /**
   * Mutex to sync the pages map
   */
  std::mutex m;

  /**
   * the buffer manager to get the pages
   */
  PDBBufferManagerInterfacePtr bufferManager;

  /**
   * Tells us what is the next page we are supposed to give a worker
   */
   std::unordered_map<size_t, pageIterator> nextPageForWorker;

  /**
   * The order in which we serve the pages
   */
  PDBAnonymousPageSetAccessPattern accessPattern = PDBAnonymousPageSetAccessPattern::SEQUENTIAL;
};



}

#endif //PDB_PDBANONYMOUSPAGESET_H
