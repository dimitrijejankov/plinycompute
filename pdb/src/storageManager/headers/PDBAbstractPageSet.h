//
// Created by dimitrije on 3/5/19.
//

#ifndef PDB_ABSTRATCTPAGESET_H
#define PDB_ABSTRATCTPAGESET_H

#include <PDBPageHandle.h>

namespace pdb {

class PDBAbstractPageSet;
using PDBAbstractPageSetPtr = std::shared_ptr<PDBAbstractPageSet>;

class PDBAbstractPageSet {
public:

  /**
   * Gets the next page in the page set
   * @param workerID - in the case that the next page is going to depend on the worker we need to specify an id for it
   * @return - page handle if the next page exists, null otherwise
   */
  virtual PDBPageHandle getNextPage(size_t workerID) = 0;

  /**
   * Creates a new page in this page set
   * @return the page handle to that page set
   */
  virtual PDBPageHandle getNewPage() = 0;

  /**
   * Return the number of pages in this page set
   * @return the number
   */
  virtual size_t getNumPages() = 0;

  /**
   * Resets the page set so it can be reused
   */
  virtual void resetPageSet() = 0;

  /**
   * Increases the number of records in the page set
   * @param numRecords - the number we want to increase by
   */
  void increaseRecords(uint64_t value) { this->numRecords += value; }

  /**
   * Decreases the number of records in the page set
   * @param numRecords - the number we want to decrease by
   */
  void decreaseRecords(uint64_t value) { this->numRecords -= value; }

  /**
   * Returns the number of records in the page set
   * @return the number of records
   */
  uint64_t getNumRecords() { return numRecords; }

protected:

  /**
   * The number of records in this page set, we don't keep the size since pages can be frozen
   */
  atomic_uint_least64_t numRecords = 0;

};

}

#endif //PDB_ABSTRATCTPAGESET_H
