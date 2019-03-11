//
// Created by dimitrije on 3/5/19.
//

#ifndef PDB_PDBSETPAGESET_H
#define PDB_PDBSETPAGESET_H

#include "PDBAbstractPageSet.h"
#include <PDBBufferManagerInterface.h>
#include <vector>

namespace pdb {

class PDBStorageManagerBackend;

// make a ptr for this type of page set
class PDBSetPageSet;
using PDBSetPageSetPtr = std::shared_ptr<PDBSetPageSet>;

class PDBSetPageSet : public PDBAbstractPageSet {
public:

  /**
   * Initializes the page set with the set parameters and the buffer manager.
   * The buffer manager is used to grab pages from the frontend.
   * @param db - the name of the database the set belongs to
   * @param set - the set name
   * @param numPages - the number of pages
   * @param bufferManager - the buffer manager
   */
  PDBSetPageSet(const std::string &db, const std::string &set, size_t numPages, PDBBufferManagerInterfacePtr bufferManager);

  /**
   * Grabs the next page for this set.
   * @param workerID - the worker id does nothing in this case
   * @return the page handle if there is one, null otherwise
   */
  PDBPageHandle getNextPage(size_t workerID) override;

  /**
   * Creates a new page in this page set by contacting the buffer manager // TODO this should probably contact the storage manager
   * @return - the page handle of the newly created page
   */
  PDBPageHandle getNewPage() override;

  /**
   * Return the number of pages in this page set
   * @return - that number
   */
  size_t getNumPages() override;

 private:

  // current page, it is thread safe to update it
  atomic_uint64_t curPage;

  // last page, it is thread safe
  atomic_uint64_t lastPage;

  // the database the set this page set corresponds to belongs
  std::string db;

  // the name of the set this page set corresponds to
  std::string set;

  // the page handles
  std::vector<PDBPageHandle> pages;

  // the buffer manager to get the pages
  PDBBufferManagerInterfacePtr bufferManager;
};

}

#endif //PDB_PDBSETPAGESET_H
