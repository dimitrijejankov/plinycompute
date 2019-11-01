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
  std::vector<Handle<OutputType>> *inputBuffer = nullptr;

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
    *lhs = rhs;
  }

  // used to to create a new key or value if needed
  template<typename T>
  typename std::enable_if<!std::is_base_of<HandleBase, T>::value, void>::type
  inline create(T &) {}

  // used to to create a new key or value if needed
  template<typename T>
  typename std::enable_if<std::is_base_of<HandleBase, T>::value, void>::type
  inline create(T &val) {

    // make the value
    val = makeObject<typename remove_handle<T>::type>();
  }

public:

  // the first param is a callback function that the iterator will call in order to obtain another vector
  // to iterate over.  The second param tells us how many objects to put into a tuple set
  MapTupleSetIterator(const PDBAbstractPageSetPtr &pageSet, uint64_t workerID, size_t chunkSize) {

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
    output->addColumn(0, new std::vector<Handle<OutputType>>, true);
  }

  ~MapTupleSetIterator() override {
    delete inputBuffer;
  };

  // returns the next tuple set to process, or nullptr if there is not one to process
  TupleSetPtr getNextTupleSet(const PDBTupleSetSizePolicy &policy) override {

    /**
     * 0. In case of failure we need to reprocess the input, copy the current stuff into the buffer
     */

    // did we manage to process the input, if not move the records into the buffer
    if(!policy.inputWasProcessed() && output != nullptr && inputBuffer != nullptr) {

      // get the input column
      std::vector<Handle<OutputType>> &inputColumn = output->getColumn<Handle<OutputType>>(0);

      // if the buffer is empty we just swap this is an optimization since that means we are not doing a copy
      if(inputBuffer->empty()) {
        std::swap(inputColumn, *inputBuffer);
      } else {

        // copy the input column and clear it
        inputBuffer->insert(inputBuffer->end(), inputColumn.begin(), inputColumn.end());
        inputBuffer->clear();
      }
    }

    /**
     * 1. We need to check if the buffer has something, if it does we need to process it.
     */

    if(inputBuffer != nullptr && !inputBuffer->empty()) {

      // get the input column
      std::vector<Handle<Object>> &inputColumn = output->getColumn<Handle<Object>>(0);

      // figure out the number to move
      auto numToCopy = std::min((size_t) policy.getChunksSize(), inputBuffer->size());

      // move the stuff out of the buffer
      inputColumn.clear();
      inputColumn.insert(inputColumn.end(),inputBuffer->end() - numToCopy,inputBuffer->end());
      inputBuffer->resize(inputBuffer->size() - numToCopy);

      // return the output
      return output;
    }

    /**
     * 2. We need to grab our tupleSet from the page, we do here a bunch of checking to know from what page we
     *    need to grab the records.
     */

    std::vector<Handle<OutputType>> &inputColumn = output->getColumn<Handle<OutputType>>(0);
    int limit = (int) inputColumn.size();

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

    for (int i = 0; i < policy.getChunksSize(); i++) {

      if (i >= limit) {

        // make the object
        Handle<OutputType> temp = (makeObject<OutputType>());

        // make the key and value if needed
        create(temp->getKey());
        create(temp->getValue());

        // push the column
        inputColumn.push_back(temp);
      }

      // key the key/value pair
      assign(inputColumn[i]->getKey(), (*begin).key);
      assign(inputColumn[i]->getValue(), (*begin).value);

      // move on to the next item
      ++begin;

      // and exit if we are done
      if (!(begin != end)) {

        if (i + 1 < limit) {
          inputColumn.resize(i + 1);
        }
        return output;
      }
    }

    return output;
  }


};

}