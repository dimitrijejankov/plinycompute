#include <utility>
#include <PDBAbstractPageSet.h>
#include <ComputeSource.h>
#include <JoinPlannerResult.h>
#include <PDBRandomAccessPageSet.h>
#include "../../../../applications/TestConvolution/sharedLibraries/headers/MatrixBlock3D.h"

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

using namespace pdb::matrix_3d;

namespace pdb {

/**
 * This class iterates over an input pdb::Vector, breaking it up into a series of TupleSet objects
 */
class Join8MainSource : public ComputeSource {

 private:

  // the page set over which we are iterating
  PDBRandomAccessPageSetPtr pageSet;

  // the id of the worker that is iterating over the page set
  uint64_t workerID;

  // this is the vector to process
  std::vector<Handle<Vector<Handle<MatrixBlock3D>>>> iterateOverThese;

  // the pages from the page set
  std::vector<PDBPageHandle> storedPage;

  // and the tuple set we return
  TupleSetPtr output;

  // the tid mapping to a page
  const std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> &tidToRecordMapping;

  const std::vector<EightWayJoinPipeline::joined_record> &joinedRecords;

  // the start and the end join groups
  uint64_t joinGroupStart;
  uint64_t joinGroupEnd;

  // the current record
  uint64_t currentRecord;

  // the input column of this source
  std::vector<Handle<MatrixBlock3D>> *inputColumn;

public:

 /**
  * Initializes the Vector8MainSource with a page set from which we are going to grab the pages from.
  *
  * @param pageSetIn - the page set we are going to grab the pages from
  * @param chunkSize - the chunk size tells us how many objects to put into a tuple set
  * @param workerID - the worker id is used a as a parameter @see PDBAbstractPageSetPtr::getNextPage to get a specific page for a worker
  */
  Join8MainSource(PDBAbstractPageSetPtr pageSetIn,
                  uint64_t workerID,
                  uint64_t numWorkers,
                  const std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> &mappings,
                  const std::vector<EightWayJoinPipeline::joined_record> &joinedRecords) : joinedRecords(joinedRecords),
                                                                                           workerID(workerID),
                                                                                           tidToRecordMapping(mappings) {
  // create the tuple set that we'll return during iteration
  output = std::make_shared<TupleSet>();

  // get the page set
  pageSet = dynamic_pointer_cast<PDBRandomAccessPageSet>(pageSetIn);

  // create the output vector and put it into the tuple set
  inputColumn = new std::vector<Handle<MatrixBlock3D>>;
  output->addColumn(0, inputColumn, true);

  // set the current page (can be null if there is none)
  for(int i = 0; i < pageSet->getNumPages(); ++i) {

    // get the page
    auto curPage = (*pageSet)[i];

    // repin the page
    curPage->repin();

    // get the record from the page
    auto curRec = (Record<Vector<Handle<MatrixBlock3D>>> *) curPage->getBytes();

    // get the root object of the page
    auto iterateOverMe = curRec->getRootObject();

    // insert the vector from the page
    iterateOverThese.push_back(iterateOverMe);

    // store the page
    storedPage.push_back(curPage);
  }

  // how much per worker
  auto stride = (int32_t) joinedRecords.size() / numWorkers;

  // figure out the start and the end
  joinGroupStart = workerID * stride;
  joinGroupEnd = (workerID + 1) * stride;

  // if this is the last worker
  joinGroupEnd += workerID == (numWorkers - 1) ? (int32_t) joinedRecords.size()% numWorkers : 0;

  // we start from record zero
  currentRecord = 0;
 }

  ~Join8MainSource() override {

    // clear the pages
    storedPage.clear();
    iterateOverThese.clear();
  }

  // helper functions to find a record
  std::tuple<uint32_t, uint32_t> findRecord(uint32_t tid) {

    for (auto &m : tidToRecordMapping) {
      auto it = m.find(tid);
      if(it != m.end()){
        return it->second;
      }
    }

    throw runtime_error("This is bad we did not find a left record");
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
    if(!policy.inputWasProcessed() && output != nullptr) {
      joinGroupStart -= inputColumn->size();
    }

    // figure out the offset
    auto start = joinGroupStart + currentRecord;
    auto end = std::min(joinGroupEnd, start + policy.getChunksSize());

    // if we are done here return null
    if (start >= end) {
      return nullptr;
    }

    // the number of rows we have
    auto numRows = end - start;

    // resize the columns
    inputColumn->resize(numRows);

    // fill it up
    for (auto i = start; i < end; ++i) {

      // get the tids
      auto tid = (joinedRecords)[i].first;

      // find all the left ones we ned to join
      auto [pageID, recordID] = findRecord(tid);

      // set the column
      (*inputColumn)[i - start] = (*iterateOverThese[pageID])[recordID];
    }

    // move the current record
    currentRecord += numRows;

    // and return the output TupleSet
    return output;
  }

};

}