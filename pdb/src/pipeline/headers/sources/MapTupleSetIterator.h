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

  // the number of items to put in each chunk that we produce
  size_t chunkSize;

  // the tuple set we return
  TupleSetPtr output;

  // the iterator for the map
  PDBMapIterator<KeyType, ValueType> begin;
  PDBMapIterator<KeyType, ValueType> end;

  // the page that contains the map
  PDBPageHandle page;

 public:

  // the first param is a callback function that the iterator will call in order to obtain another vector
  // to iterate over.  The second param tells us how many objects to put into a tuple set
  MapTupleSetIterator(const PDBAbstractPageSetPtr &pageSet, uint64_t workerID, size_t chunkSize) : chunkSize(chunkSize) {

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

  // returns the next tuple set to process, or nullptr if there is not one to process
  TupleSetPtr getNextTupleSet() override {

    // do we even have a map
    if(iterateOverMe == nullptr) {
      return nullptr;
    }

    // see if there are no more items in the vector to iterate over
    if (!(begin != end)) {

      // unpin the page
      page->unpin();

      // finish
      return nullptr;
    }

    std::vector<Handle<OutputType>> &inputColumn = output->getColumn<Handle<OutputType>>(0);
    int limit = (int) inputColumn.size();
    for (int i = 0; i < chunkSize; i++) {

      if (i >= limit) {
        Handle<OutputType> temp = (makeObject<OutputType>());
        inputColumn.push_back(temp);
      }

      // key the key/value pair
      inputColumn[i]->getKey() = (*begin).key;
      inputColumn[i]->getValue() = (*begin).value;

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

  ~MapTupleSetIterator() override = default;
};

}