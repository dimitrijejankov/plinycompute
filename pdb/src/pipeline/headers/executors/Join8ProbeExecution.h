#pragma once

#include <TupleSetMachine.h>
#include <TupleSpec.h>
#include <JoinTuple.h>
#include <thread>
#include <utility>
#include "ComputeExecutor.h"
#include "StringIntPair.h"
#include "JoinMap.h"
#include "PDBRandomAccessPageSet.h"
#include "EightWayJoinPipeline.h"
#include "PDBAbstractPageSet.h"
#include "../../../../../applications/TestConvolution/sharedLibraries/headers/MatrixBlock3D.h"

using namespace pdb::matrix_3d;

namespace pdb {

// this class is used to encapsulte the computation that is responsible for probing a hash table
class Join8ProbeExecution : public ComputeExecutor {
private:

  // this is the output TupleSet that we return
  TupleSetPtr output;

  std::vector<std::vector<Handle<MatrixBlock3D>>> columns;

  int position;

  int numWorkers;

  // the page set over which we are iterating
  PDBRandomAccessPageSetPtr pageSet;

  // the id of the worker that is iterating over the page set
  uint64_t workerID;

  // this is the vector to process
  std::vector<Handle<Vector<Handle<MatrixBlock3D>>>> iterateOverThese;

  // the pages from the page set
  std::vector<PDBPageHandle> storedPage;

  // joined records
  const std::vector<EightWayJoinPipeline::joined_record> &joinedRecords;

  // the tid mapping to a page
  const std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> &tidToRecordMapping;

  // the start and the end join groups
  uint64_t joinGroupStart;
  uint64_t joinGroupEnd;

  // the current record
  uint64_t currentRecord;

  uint64_t numRecords;

 public:

  ~Join8ProbeExecution() = default;

  // when we probe a hash table, a subset of the atts that we need to put into the output stream are stored in the hash table... the positions
  // of these packed atts are stored in typesStoredInHash, so that they can be extracted.  inputSchema, attsToOperateOn, and attsToIncludeInOutput
  // are standard for executors: they tell us the details of the input that are streaming in, as well as the identity of the has att, and
  // the atts that will be streamed to the output, from the input.  needToSwapLHSAndRhs is true if it's the case that the atts stored in the
  // hash table need to come AFTER the atts being streamed through the join

  Join8ProbeExecution(const PDBAbstractPageSetPtr& hashTable,
                      int position,
                      uint64_t workerID,
                      int numWorkers,
                      const std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> &tidToRecordMapping,
                      const std::vector<EightWayJoinPipeline::joined_record> &joinedRecords) : position(position),
                                                                                               workerID(workerID),
                                                                                               numWorkers(numWorkers),
                                                                                               tidToRecordMapping(tidToRecordMapping),
                                                                                               joinedRecords(joinedRecords) {

    // make the output
    output = std::make_shared<TupleSet>();

    // get the page set
    pageSet = dynamic_pointer_cast<PDBRandomAccessPageSet>(hashTable);

    // insert the columns
    columns.resize(position + 1);
    for(int i = 0; i < position + 1; ++i) {
      output->addColumn(i, &columns[i], false);
    }

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
      iterateOverThese.emplace_back(iterateOverMe);

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

  void print_first(){

    std::cout << "----------------------\n";
    for(int i = 0; i <= position; ++i) {
      auto handle = columns[i][0];

      std::cout << handle->getKey()->x_id << " " <<  handle->getKey()->y_id << " " <<  handle->getKey()->z_id << '\n';
    }
    std::cout << "----------------------\n";

  }

  void markAsProcessed() override {

    // move the current record
    currentRecord += numRecords;
  }

  TupleSetPtr process(TupleSetPtr input) override {

    for(int i = 0; i < position; ++i) {

      // get the column
      auto column = &input->getColumn<Handle<MatrixBlock3D>>(i);

      // copy the column
      columns[i] = *column;
    }

    // how many records are there
    numRecords = columns[position - 1].size();

    // resize
    columns[position].resize(numRecords);

    // copy all the necessary records
    for(int i = 0; i < numRecords; ++i) {

      // figure out the tid
      int32_t tid;
      switch (position) {
        case 0: { tid = (joinedRecords)[currentRecord + i].first; break; }
        case 1: { tid = (joinedRecords)[currentRecord + i].second; break; }
        case 2: { tid = (joinedRecords)[currentRecord + i].third; break; }
        case 3: { tid = (joinedRecords)[currentRecord + i].fourth; break; }
        case 4: { tid = (joinedRecords)[currentRecord + i].fifth; break; }
        case 5: { tid = (joinedRecords)[currentRecord + i].sixth; break; }
        case 6: { tid = (joinedRecords)[currentRecord + i].seventh; break; }
        case 7: { tid = (joinedRecords)[currentRecord + i].eight; break; }
        default: { throw runtime_error("Not good!"); }
      }

      // find all the left ones we ned to join
      auto [pageID, recordID] = findRecord(tid);

      // set the column
      columns[position][i] = (*iterateOverThese[pageID])[recordID];
    }

    //print_first();

    // outta here!
    return output;
  }
};

}