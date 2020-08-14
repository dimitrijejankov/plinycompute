//
// Created by dimitrije on 3/5/19.
//

#ifndef PDB_ABSTRATCTPAGESET_H
#define PDB_ABSTRATCTPAGESET_H

#include <PDBPageHandle.h>
#include <PDBString.h>


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
   * Removes a page from the page set
   * @param pageHandle - the page set
   */
  virtual void removePage(PDBPageHandle pageHandle) = 0;

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
   * Returns the maximum page size this page set can give
   * @return the maximum page size, if method is defined
   */
  virtual size_t getMaxPageSize() = 0;

  /**
   * Increases the number of records in the page set
   * @param numRecords - the number we want to increase by
   */
  void increaseRecords(uint64_t value) { this->numRecords += value; }

  /**
   * Increases the size stat of the page set
   * @param value - the number we want to increase by
   */
  void increaseSize(uint64_t value) { this->size += value; }

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

  /**
   * Returns the size of the page set
   * @return the size
   */
  [[nodiscard]] uint64_t getSize() const { return size; }

  /**
   * Converts a page set identifier to it's key version. We use this method for consistency
   * @param pageSetID - the identifier of the page set
   * @return the the identifier of page set that stores the keys
   */
  static auto toKeyPageSetIdentifier(const std::pair<size_t, pdb::String> &pageSetID) {
    return std::make_pair(pageSetID.first, (std::string) pageSetID.second + "#keys");
  }

  /**
 * Converts a page set identifier to it's key version. We use this method for consistency
 * @param pageSetID - the identifier of the page set
 * @return the the identifier of page set that stores the keys
 */
  static auto toKeyPageSetIdentifier(const std::pair<size_t, pdb::string> &pageSetID) {
    return std::make_pair(pageSetID.first, (std::string) pageSetID.second + "#keys");
  }

protected:

  /**
   * The number of records in this page set, we don't keep the size since pages can be frozen
   */
  atomic_uint_least64_t numRecords = 0;

  /**
   * Keeps track of the size of the page set
   */
  atomic_uint_least64_t size = 0;

};

}

#endif //PDB_ABSTRATCTPAGESET_H
