#include <utility>

#pragma once

#include <ComputeSource.h>
#include <TupleSpec.h>
#include <TupleSetMachine.h>
#include <JoinTuple.h>

namespace pdb {

template<typename RHS>
class LHSShuffleJoinSource : public ComputeSource {
 private:

  // and the tuple set we return
  TupleSetPtr output;

  // tells us which attribute is the key
  int keyAtt;

  // the attribute order of the records
  std::vector<int> recordAttributes;

  // the vector that contains the hash column
  std::vector<size_t> hashColumn;

  // to setup the output tuple set
  TupleSetSetupMachine myMachine;

  // the page set we are going to be grabbing the pages from
  PDBAbstractPageSetPtr pageSet;

  // the current page we are reading from
  PDBPageHandle currPage = nullptr;

  // the previous page
  PDBPageHandle prevPage = nullptr;

  // this is the current map
  Handle<JoinMap<RHS>> currMap = nullptr;

  // the current iterator
  JoinMapIterator<RHS> currentIt;

  // the number of tuples in the tuple set
  int32_t chunkSize = 0;

  // this keeps track of the last object in the map we accessed by the previous call of getNextTupleSet
  int32_t lastObject = 0;

  // this is the worker we are doing the processing for
  uint64_t workerID = 0;

  // the output columns of the tuple set
  void **columns;

 public:

  LHSShuffleJoinSource(TupleSpec &inputSchema,
                       TupleSpec &hashSchema,
                       TupleSpec &recordSchema,
                       std::vector<int> &recordOrder,
                       PDBAbstractPageSetPtr leftInputPageSet,
                       int32_t chunkSize,
                       uint64_t workerID)
      : myMachine(inputSchema), pageSet(std::move(leftInputPageSet)), chunkSize(chunkSize), workerID(workerID) {

    // create the tuple set that we'll return during iteration
    output = std::make_shared<TupleSet>();

    // figure out the key att
    std::vector<int> matches = myMachine.match(hashSchema);
    keyAtt = matches[0];

    // figure the record attributes
    recordAttributes = myMachine.match(recordSchema);

    // allocate a vector for the columns
    columns = new void *[recordAttributes.size()];

    // create the columns for the records
    createCols<RHS>(columns, *output, 0, 0, recordOrder);

    // add the hash column
    output->addColumn(keyAtt, &hashColumn, false);

    // try to find a map
    getNextMap();
  }

  ~LHSShuffleJoinSource() override {

    // unpin the last previous page
    if (prevPage != nullptr) {
      prevPage->unpin();
    }

    // delete the columns
    delete[] columns;
  }

  TupleSetPtr getNextTupleSet() override {

    // if we dom't have any pages finish
    if (currPage == nullptr) {
      return nullptr;
    }

    // fill up the output
    int count = 0;
    while (currentIt != currMap->end()) {

      // just to make the code look nicer
      auto tmp = *currentIt;
      auto &currentRecords = *tmp;

      // get the hash of the iterator
      size_t hash = currentRecords.getHash();

      // fill up the output
      for (; lastObject < currentRecords.size(); ++lastObject) {

        // unpack the record
        unpack(currentRecords[lastObject], count++, 0, columns);
        hashColumn.emplace_back(hash);

        // did we fill the output, if so return the tuple set
        if (count >= chunkSize) {
          return output;
        }
      }

      // reset the counter
      lastObject = 0;

      // go to the next iterator
      currentIt.operator++();
    }

    // truncate if we have extra
    eraseEnd<RHS>(count, 0, columns);
    hashColumn.resize((unsigned) count);

    // try to find a map
    getNextMap();

    return output;
  }

  void getNextMap() {

    // while we don't find a good page
    do {

      // unpin the previous page if any
      if (prevPage != nullptr) { prevPage->unpin(); }

      //
      prevPage = currPage;

      // get the next page
      currPage = pageSet->getNextPage(0);

      // if we did not get a page, finish there is not next map
      if (currPage == nullptr) {
        break;
      }

      // repin the next page
      currPage->repin();

      // we grab the vector of hashmaps
      Handle<Vector<Handle<JoinMap<RHS>>>>
          returnVal = ((Record<Vector<Handle<JoinMap<RHS>>>> *) (currPage->getBytes()))->getRootObject();

      // next we grab the join map we need
      currMap = (*returnVal)[workerID];

      // grab the current iterator
      currentIt = currMap->begin();

      // do this while we don't get a map with stuff on it!
    } while (currMap->size() == 0);

  }

};

}
