#pragma once

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

  PDBStorageIterator() = default;

  /**
   * Checks if there is another record that we haven't visited
   * @return true if there is false otherwise
   */
  virtual bool hasNextRecord() = 0;

  /**
   * Returns the next record.
   * @return returns the record if there is one nullptr otherwise
   */
  virtual pdb::Handle<T> getNextRecord() = 0;

};

}
