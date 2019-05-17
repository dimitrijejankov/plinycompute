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

 private:

  /**
   * Keeps track of all anonymous pages so that we can quickly remove them
   */
  std::map<uint64_t, PDBPageHandle> pages;

  /**
   * The current page when iterating
   */
  std::map<uint64_t, PDBPageHandle>::iterator curPage;

  /**
   * Indicates whether we are done iterating
   */
  bool isDone;

  /**
   * Indicates whether we need to initialize the page set iterator
   */
  bool needsInitialization;

  /**
   * Mutex to sync the pages map
   */
  std::mutex m;

  /**
   * the buffer manager to get the pages
   */
  PDBBufferManagerInterfacePtr bufferManager;

};



}

#endif //PDB_PDBANONYMOUSPAGESET_H
