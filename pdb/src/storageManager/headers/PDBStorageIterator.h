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
   * The buffer we are storing the records
   */
  std::unique_ptr<char[]> buffer;

  /**
   * The size of the buffer
   */
  size_t bufferSize = 0;

  /**
   * The set this iterator belongs to
   */
  std::string set;

  /**
   * The database the set belongs to
   */
  std::string db;

  /**
   * The node we want to grab the page from
   */
  std::string currNode;

  /**
   * The number of the page we want to get
   */
  int64_t  currPage = -1;

  /**
   * The current record on the page
   */
  int64_t currRecord = -1;
};

}

// include the implementation
#include "PDBStorageIteratorTemplate.cc"

#endif //PDB_STORAGEITERATOR_H
