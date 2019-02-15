//
// Created by dimitrije on 2/13/19.
//

#ifndef PDB_STORAGEITERATOR_H
#define PDB_STORAGEITERATOR_H

#include <string>
#include <memory>
#include <Handle.h>

namespace pdb {


template <class T>
class PDBStorageIterator;

template <class T>
using PDBStorageIteratorPtr = std::shared_ptr<PDBStorageIterator<T>>;

template <class T>
class PDBStorageIterator {

public:

  PDBStorageIterator(const std::string &address,
                     int port,
                     int maxRetries,
                     const std::string &set,
                     const std::string &db);


  /**
   * Checks if there is another record that we haven't visited
   * @return true if there is false otherwise
   */
  bool hasNextRecord();

  /**
   * Returns the next record.
   * @return returns the record if there is one nullptr otherwise
   */
  pdb::Handle<T> getNextRecord();

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
#include "PDBStorageIteratorTemplate.cc"

#endif //PDB_STORAGEITERATOR_H
