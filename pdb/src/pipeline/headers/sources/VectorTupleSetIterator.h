#include <utility>
#include <PDBAbstractPageSet.h>
#include <ComputeSource.h>

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

#ifndef VECTOR_TUPLESET_ITER_H
#define VECTOR_TUPLESET_ITER_H

namespace pdb {

/**
 * This class iterates over an input pdb::Vector, breaking it up into a series of TupleSet objects
 */
class VectorTupleSetIterator : public ComputeSource {

 private:

  // the page set over which we are iterating
  PDBAbstractPageSetPtr pageSet;

  // the id of the worker that is iterating over the page set
  uint64_t workerID;

  // the page we are currently iterating over
  PDBPageHandle curPage;

  // the page we were using before
  PDBPageHandle lastPage;

  // this is the vector to process
  Handle<Vector<Handle < Object>>> iterateOverMe;

  // the pointer to the current page holding the vector
  Record<Vector<Handle < Object>>> *curRec;

  // the pointer to holding the last page that we previously processed
  Record<Vector<Handle < Object>>> *lastRec;

  // where we are in the chunk
  size_t pos;

  // and the tuple set we return
  TupleSetPtr output;

  // the buffer where we put records in the case of a failed processing attempt
  std::vector<Handle<Object>> *inputBuffer = nullptr;

public:

 /**
  * Initializes the VectorTupleSetIterator with a page set from which we are going to grab the pages from.
  *
  * @param pageSetIn - the page set we are going to grab the pages from
  * @param chunkSize - the chunk size tells us how many objects to put into a tuple set
  * @param workerID - the worker id is used a as a parameter @see PDBAbstractPageSetPtr::getNextPage to get a specific page for a worker
  */
  VectorTupleSetIterator(PDBAbstractPageSetPtr pageSetIn, size_t chunkSize, uint64_t workerID) : pageSet(std::move(pageSetIn)), workerID(workerID) {

    // create the tuple set that we'll return during iteration
    output = std::make_shared<TupleSet>();

    // set the current page (can be null if there is none)
    curPage = pageSet->getNextPage(workerID);

    // check if we actually have a page
    if(curPage == nullptr) {

      // set the current rec to null
      curRec = nullptr;
      lastRec = nullptr;
      pos = 0;

      // just get out
      return ;
    }

    // repin the page
    curPage->repin();

    // extract the vector from the first page if there is no page just set it to null
    curRec = (Record<Vector<Handle<Object>>> *) curPage->getBytes();
    if (curRec != nullptr) {

      // get the root object of the page
      iterateOverMe = curRec->getRootObject();

      // create the output vector and put it into the tuple set
      auto *inputColumn = new std::vector<Handle<Object>>;
      output->addColumn(0, inputColumn, true);

      // initialize the buffer
      inputBuffer = new std::vector<Handle<Object>>;

    } else {

      iterateOverMe = nullptr;
      output = nullptr;
    }

    // we are at position zero
    pos = 0;

    // and we have no data so far
    lastRec = nullptr;
  }

  ~VectorTupleSetIterator() override {

    // set the last rec to null
    lastRec = nullptr;

    // delete the input buffer
    delete inputBuffer;
  }

  /**
   * returns the next tuple set to process, or nullptr if there is not one to process
   * @return - the mentioned tuple set
   */
  TupleSetPtr getNextTupleSet(const PDBTupleSetSizePolicy &policy) override {

    /**
     * 0. In case of failure we need to reprocess the input, copy the current stuff into the buffer
     */

    // did we manage to process the input, if not move the records into the buffer
    if(!policy.inputWasProcessed() && output != nullptr && inputBuffer != nullptr) {

      // get the input column
      std::vector<Handle<Object>> &inputColumn = output->getColumn<Handle<Object>>(0);

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

    // if we made it here with lastRec being a valid pointer, then it means
    // that we have gone through an entire cycle, and so all of the data that
    // we will ever reference stored in lastRec has been flushed through the
    // pipeline; hence, we can kill it
    if (lastRec != nullptr) {

      // kill the page
      lastPage->unpin();
      lastPage = nullptr;

      // kill the record
      lastRec = nullptr;
    }

    // if we did not get a page we don't have any records..
    if(curPage == nullptr) {
      return nullptr;
    }

    // see if there are no more items in the vector to iterate over
    if (pos == iterateOverMe->size()) {

      // this means that we got to the end of the vector
      lastRec = curRec;
      lastPage = curPage;

      // try to get another vector
      curPage = pageSet->getNextPage(workerID);

      // if we could not, then we are outta here
      if (curPage == nullptr) {
        return nullptr;
      }

      // repin the page
      curPage->repin();

      // extract the vector from the first page if there is no page just set it to null
      curRec = (Record<Vector<Handle<Object>>> *) curPage->getBytes();

      // and reset everything
      iterateOverMe = curRec->getRootObject();
      pos = 0;
    }

    /**
     * 3. Here we figure out how many records we need to grab. And then just copy.
     */

    // compute how many slots in the output vector we can fill
    size_t numSlotsToIterate = policy.getChunksSize();
    if (numSlotsToIterate + pos > iterateOverMe->size()) {
      numSlotsToIterate = iterateOverMe->size() - pos;
    }

    // resize the output vector as appropriate
    std::vector<Handle<Object>> &inputColumn = output->getColumn<Handle<Object>>(0);
    inputColumn.resize(numSlotsToIterate);

    // fill it up
    for (int i = 0; i < numSlotsToIterate; i++) {
      inputColumn[i] = (*iterateOverMe)[pos];
      pos++;
    }

    // and return the output TupleSet
    return output;
  }

};

}

#endif
