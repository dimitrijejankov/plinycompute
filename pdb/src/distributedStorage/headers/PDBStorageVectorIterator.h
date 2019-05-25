#pragma once

#include "PDBStorageIterator.h"
#include <string>

namespace pdb {

template <class T>
class PDBStorageVectorIterator : public PDBStorageIterator<T> {
public:


  PDBStorageVectorIterator(std::string address, int port, int maxRetries, std::string set, std::string db);


  /**
   * Checks if there is another record that we haven't visited
   * @return true if there is false otherwise
   */
  bool hasNextRecord() override;

  /**
   * Returns the next record.
   * @return returns the record if there is one nullptr otherwise
   */
  pdb::Handle<T> getNextRecord() override;

private:

  /**
   * Grab the next page
   * @return true if we could grab the next page
   */
  bool getNextPage(bool isFirst);

  /**
   * the address of the manager
   */
  std::string address;

  /**
   * the port of the manager
   */
  int port = -1;

  /**
   * How many times should we retry to connect to the manager if we fail
   */
  int maxRetries = 1;

  /**
   * the logger
   */
  PDBLoggerPtr logger;

  /**
   * The set this iterator belongs to
   */
  std::string set;

  /**
   * The database the set belongs to
   */
  std::string db;

  /**
   * The number of the page we want to get
   */
  uint64_t currPage = 0;

  /**
   * The node we want to grab the page from
   */
  std::string currNode = "none";

  /**
   * The current record on the page
   */
  int64_t currRecord = -1;

  /**
   * The buffer we are storing the records
   */
  std::unique_ptr<char[]> buffer;

  /**
   * The size of the buffer
   */
  size_t bufferSize = 0;
};

}

// include the implementation
#include "PDBStorageVectorIteratorTemplate.cc"