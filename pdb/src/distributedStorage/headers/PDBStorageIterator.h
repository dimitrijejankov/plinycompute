#pragma once

#include <string>
#include <memory>
#include <Handle.h>
#include <PDBConnectionManager.h>

namespace pdb {


template <class T>
class PDBStorageIterator;

template <class T>
using PDBStorageIteratorPtr = std::shared_ptr<PDBStorageIterator<T>>;

template <class T>
class PDBStorageIterator {

public:

  explicit PDBStorageIterator(PDBConnectionManager *conMgr) : conMgr(conMgr) {}

  // Checks if there is another record that we haven't visited
  virtual bool hasNextRecord() = 0;

  // Returns the next record.
  virtual pdb::Handle<T> getNextRecord() = 0;

protected:

  // manages the connections
  PDBConnectionManager *conMgr;
};

}
