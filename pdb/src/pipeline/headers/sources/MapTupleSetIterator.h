/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#pragma once

#include <ComputeSource.h>
#include <PDBPageHandle.h>
#include <PDBAbstractPageSet.h>

namespace pdb {

// this class iterates over a pdb :: Map, returning a set of TupleSet objects
template<typename KeyType, typename ValueType, typename OutputType>
class MapTupleSetIterator : public ComputeSource {

private:

  // the map we are iterating over
  Handle<Map<KeyType, ValueType>> iterateOverMe;

  // the tuple set we return
  TupleSetPtr output;

  // the iterator for the map
  PDBMapIterator<KeyType, ValueType> begin;
  PDBMapIterator<KeyType, ValueType> end;

  // the page that contains the map
  PDBPageHandle page;

  // the buffer where we put records in the case of a failed processing attempt
  std::vector<Handle<OutputType>> buffer;

  // here we keep the OutputType object we are emitting from the iterator as RefCountedObjects
  std::vector<int8_t> bufferStorage;

  // the buffer where we put records in the case of a failed processing attempt
  std::vector<Handle<OutputType>> column;

  // here we keep the OutputType object we are emitting from the iterator as RefCountedObjects
  std::vector<int8_t> columnStorage;

  // how much memory do we need
  const uint64_t REF_COUNTED_OBJECT_SIZE = REF_COUNT_PREAMBLE_SIZE + sizeof(OutputType);

  // used to assign rhs to the lhs if the lhs is not a handle
  template<typename LHS, typename RHS>
  typename std::enable_if<!std::is_base_of<HandleBase, LHS>::value, void>::type
  inline assign(LHS &lhs, RHS &rhs) {
    lhs = rhs;
  }

  // used to assign rhs to the lhs if the lhs is a handle
  template<typename LHS, typename RHS>
  typename std::enable_if<std::is_base_of<HandleBase, LHS>::value, void>::type
  inline assign(LHS &lhs, RHS &rhs) {
    lhs.setOffset(-1);
    lhs = (RefCountedObject<RHS> *)(((int8_t *) &rhs) - REF_COUNT_PREAMBLE_SIZE);
  }

  // used to allocate a buffer
  void allocateOutputBuffer(uint64_t numObjects, std::vector<Handle<OutputType>> &c, std::vector<int8_t> &storage) {

    // how much should we allocate
    storage.resize(numObjects * REF_COUNTED_OBJECT_SIZE);
    c.resize(numObjects);

    // do the default constructor here
    for(int i = 0; i < numObjects; ++i) {

      // make the object with the preamble
      *((unsigned *) &storage[i * REF_COUNTED_OBJECT_SIZE]) = 1;
      *((OutputType) &storage[i * REF_COUNTED_OBJECT_SIZE + REF_COUNT_PREAMBLE_SIZE]) = OutputType();

      // make the handle
      c[i].setOffset(-1);
      c[i] = ((RefCountedObject<OutputType>*) &storage[i * REF_COUNTED_OBJECT_SIZE]);
    }
  }

  void copyToBuffer() {

    // allocate the temporary buffer
    std::vector<int8_t> tmpBuffer((column.size() + buffer.size()) * REF_COUNTED_OBJECT_SIZE);

    // copy the buffer to the tmp
    for(int i = 0; i < buffer.size(); ++i) {

      // copy the object with the preamble
      *((unsigned *) &tmpBuffer[i * REF_COUNTED_OBJECT_SIZE]) = 1;
      *((OutputType*) &tmpBuffer[i * REF_COUNTED_OBJECT_SIZE + REF_COUNT_PREAMBLE_SIZE]) = *buffer[i];
    }

    // copy the column to the temp buffer
    for(int i = buffer.size(); i < (buffer.size() + column.size()); ++i) {

      // copy the object with the preamble
      *((unsigned *) &tmpBuffer[i * REF_COUNTED_OBJECT_SIZE]) = 1;
      *((OutputType*) &tmpBuffer[i * REF_COUNTED_OBJECT_SIZE + REF_COUNT_PREAMBLE_SIZE]) = *column[i - buffer.size()];
    }

    // copy the handles
    buffer.resize((buffer.size() + column.size()));
    for(int i = 0; i < (column.size() + column.size()); ++i) {
      buffer[i].setOffset(-1);
      buffer[i] = ((RefCountedObject<OutputType>*) &tmpBuffer[i * REF_COUNTED_OBJECT_SIZE + REF_COUNT_PREAMBLE_SIZE]);
    }

    // swap the storage
    std::swap(tmpBuffer, bufferStorage);

    // clear
    column.clear();
  }

  void fromBufferToColumn(int32_t numObjects) {

    // resize the output column if necessary
    if(columnStorage.size() < numObjects * REF_COUNTED_OBJECT_SIZE) {

      // resize the storage
      columnStorage.resize(numObjects * REF_COUNTED_OBJECT_SIZE);

      // make the columns point where they should
      for(int i = 0; i < numObjects; ++i) {
        column[i].setOffset(-1);
        column[i] = ((RefCountedObject<OutputType>*) &columnStorage[i * REF_COUNTED_OBJECT_SIZE]);
      }
    }
    column.resize(numObjects);

    // copy the buffer to the tmp
    for(int32_t i = buffer.size() - numObjects; i < buffer.size(); ++i) {

      // copy the object with the preamble
      *((unsigned *) &columnStorage[i * REF_COUNTED_OBJECT_SIZE]) = 1;
      *((OutputType*) &columnStorage[i * REF_COUNTED_OBJECT_SIZE + REF_COUNT_PREAMBLE_SIZE]) = *buffer[i];
    }

    // resize the buffer and the storage
    buffer.resize(buffer.size() - numObjects);
  }

  void resize_column_storage(int numObjects) {

    // resize the output column
    if(columnStorage.size() < numObjects * REF_COUNTED_OBJECT_SIZE) {

      // resize the storage
      columnStorage.resize(numObjects * REF_COUNTED_OBJECT_SIZE);

      // copy the buffer to the tmp
      for(int32_t i = 0; i < numObjects; ++i) {

        // copy the object with the preamble
        *((unsigned *) &columnStorage[i * REF_COUNTED_OBJECT_SIZE]) = 1;
        *((OutputType*) &columnStorage[i * REF_COUNTED_OBJECT_SIZE + REF_COUNT_PREAMBLE_SIZE]) = OutputType();
      }
    }
  }

public:

  // the first param is a callback function that the iterator will call in order to obtain another vector
  // to iterate over.  The second param tells us how many objects to put into a tuple set
  MapTupleSetIterator(const PDBAbstractPageSetPtr &pageSet, uint64_t workerID) {

    // get the page if we have one if we don't set the hash map to null
    page = pageSet->getNextPage(workerID);

    // repin the page
    page->repin();

    if(page == nullptr) {
      iterateOverMe = nullptr;
      return;
    }

    // get the hash table
    Handle<Object> myHashTable = ((Record<Object> *) page->getBytes())->getRootObject();
    iterateOverMe = unsafeCast<Map<KeyType, ValueType>>(myHashTable);

    // get the iterators
    begin = iterateOverMe->begin();
    end = iterateOverMe->end();

    // make the output set
    output = std::make_shared<TupleSet>();
    output->addColumn(0, &column, false);
  }

  ~MapTupleSetIterator() override = default;

  // returns the next tuple set to process, or nullptr if there is not one to process
  TupleSetPtr getNextTupleSet(const PDBTupleSetSizePolicy &policy) override {

    /**
     * 0. In case of failure we need to reprocess the input, copy the current stuff into the buffer
     */

    // did we manage to process the input, if not move the records into the buffer
    if(!policy.inputWasProcessed() && output != nullptr) {
      copyToBuffer();
    }

    /**
     * 1. We need to check if the buffer has something, if it does we need to process it.
     */

    if(!buffer.empty()) {

      // figure out the number to move
      auto numToCopy = std::min((size_t) policy.getChunksSize(), buffer.size());

      // move the stuff out of the buffer
      fromBufferToColumn(numToCopy);

      // return the output
      return output;
    }

    /**
     * 2. We need to grab our tupleSet from the page, we do here a bunch of checking to know from what page we
     *    need to grab the records.
     */

    // we always have enough storage allocated to fill the whole chunk
    resize_column_storage(policy.getChunksSize());

    // assume the column is full
    column.resize(policy.getChunksSize());

    // do we even have a map
    if(iterateOverMe == nullptr) {
      return nullptr;
    }

    // see if there are no more items in the map to iterate over
    if (!(begin != end)) {

      // unpin the page
      page->unpin();

      // finish
      return nullptr;
    }

    // fill up the column
    for (int i = 0; i < policy.getChunksSize(); i++) {

      // key the key/value pair
      column[i].setOffset(-1);
      column[i] = ((RefCountedObject<OutputType>*) &columnStorage[i * REF_COUNTED_OBJECT_SIZE]);

      // assign the key and value
      assign(column[i]->getKey(), (*begin).key);
      assign(column[i]->getValue(), (*begin).value);

      // move on to the next item
      ++begin;

      // and exit if we are done
      if (!(begin != end)) {

        if (i + 1 < column.size()) {

          // resize the output column
          column.resize(i + 1);
        }

        // return the output
        return output;
      }
    }

    return output;
  }


};

}