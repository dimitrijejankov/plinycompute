#pragma once

#include <memory>
#include <PDBPage.h>
#include <PDBPageHandle.h>

namespace pdb {

class PDBKeyExtractor {
public:

  virtual bool processedLast() = 0;

  virtual void extractKeys(const PDBPageHandle &inputPage, const PDBPageHandle &keyPage) = 0;

  virtual uint64_t pageSize(const PDBPageHandle &keyPage) = 0;

  virtual uint64_t numTuples(const PDBPageHandle &keyPage) = 0;
};

template<class T>
class PDBKeyExtractorImpl : public PDBKeyExtractor {
 public:

  // did we process the records from the last iteration
  bool _processedLast = false;

  // the index of the current record
  uint64_t _index = 0;

  // the key vector
  pdb::Handle<Vector<Handle<Object>>> _keyVector = nullptr;

  bool processedLast() override {
    return _processedLast;
  }

  // extracts the keys from the input page and moves them to the key vector
  void extractKeys(const PDBPageHandle &inputPage, const PDBPageHandle &keyPage) override {

    // did we finish everything the last time?
    if(!_processedLast) {

      // reset the index
      _index = 0;

      // make a new vector
      _keyVector = pdb::makeObject<Vector<Handle<Object>>>();
    }

    // mark that we have more to process now
    _processedLast = false;

    // cast the place where we copied the thing
    auto* recordCopy = (Record<Vector<Handle<T>>>*) inputPage->getBytes();

    // grab the copy of the supervisor object
    Handle<Vector<Handle<T>>> inputVector = recordCopy->getRootObject();

    // go through the input vector and extract the keys
    for(_index = 0; _index < inputVector->size(); ++_index) {

      // copy the key
      _keyVector->push_back((*inputVector)[_index]->getKey());
    }

    // we processed everything
    _processedLast = true;
  }

  // returns the current page size
  uint64_t pageSize(const PDBPageHandle &keyPage) override {
    return getRecord(_keyVector)->numBytes();
  }

  uint64_t numTuples(const PDBPageHandle &keyPage) override {

    // cast the place where we copied the thing
    return _keyVector->size();
  }
};

template<class K, class V>
class PDBMapKeyExtractorImpl : public PDBKeyExtractor {
 public:

  // did we process the records from the last iteration
  bool _processedLast = false;

  // the index of the current record
  uint64_t _index = 0;

  // the key vector
  pdb::Handle<Vector<Handle<Object>>> _keyVector = nullptr;

  bool processedLast() override {
    return _processedLast;
  }

  // extracts the keys from the input page and moves them to the key vector
  void extractKeys(const PDBPageHandle &inputPage, const PDBPageHandle &keyPage) override {

    // did we finish everything the last time?
    if(!_processedLast) {

      // reset the index
      _index = 0;

      // make a new vector
      _keyVector = pdb::makeObject<Vector<Handle<Object>>>();
    }

    // mark that we have more to process now
    _processedLast = false;

    // cast the place where we copied the thing
    auto* recordCopy = (Record<Map<K, V>>*) inputPage->getBytes();

    // grab the copy of the supervisor object
    auto inputVector = recordCopy->getRootObject();

    // go through the input vector and extract the keys
    for(auto it = inputVector->begin(); it != inputVector->end(); ++it) {

      // increment here
      _index++;

      // copy the key
      pdb::Handle<K> tmp = makeObject<K>();
      *tmp = it.operator*().key;
      _keyVector->push_back(tmp);
    }

    // we processed everything
    _processedLast = true;
  }

  // returns the current page size
  uint64_t pageSize(const PDBPageHandle &keyPage) override {
    return getRecord(_keyVector)->numBytes();
  }

  uint64_t numTuples(const PDBPageHandle &keyPage) override {

    // cast the place where we copied the thing
    return _keyVector->size();
  }
};

//
using PDBKeyExtractorPtr = std::shared_ptr<PDBKeyExtractor>;

}