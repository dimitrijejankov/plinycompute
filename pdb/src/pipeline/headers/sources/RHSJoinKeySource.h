#pragma once

#include <utility>

#include "RHSKeyJoinSourceBase.h"

namespace pdb {

template<typename RHS>
class RHSJoinKeySource : public RHSKeyJoinSourceBase {
public:

  RHSJoinKeySource(TupleSpec &inputSchema,
                   TupleSpec &hashSchema,
                   TupleSpec &recordSchema,
                   std::vector<int> &recordOrder,
                   PDBAbstractPageSetPtr  rightInputPageSet,
                   PDBPageHandle rightMap) : myMachine(inputSchema),
                                             pageSet(std::move(rightInputPageSet)),
                                             rightMap(std::move(rightMap)) {

    // create the tuple set that we'll return during iteration
    this->output = std::make_shared<TupleSet>();

    // figure out the key att
    std::vector<int> matches = this->myMachine.match(hashSchema);
    this->keyAtt = matches[0];

    // figure the record attributes
    this->recordAttributes = this->myMachine.match(recordSchema);

    // allocate a vector for the columns
    this->columns = new void *[this->recordAttributes.size()];

    // create the columns for the records
    createCols<RHS>(this->columns, *this->output, 0, 0, recordOrder);

    // add the hash column
    this->output->addColumn(this->keyAtt, &this->hashColumn, false);
  }

  ~RHSJoinKeySource() override {
    delete[] columns;
  }

  void initialize() {

    PDBPageHandle page;
    while ((page = this->pageSet->getNextPage(0)) != nullptr) {

      // pin the page
      page->repin();

      // we grab the vector of hash maps
      Handle<JoinMap<RHS>> returnVal = ((Record<JoinMap<RHS>> *) (page->getBytes()))->getRootObject();

      // next we grab the join map we need
      this->maps.push_back(returnVal);

      // if the map has stuff add it to the queue
      auto it = this->maps.back()->begin();
      if (it != this->maps.back()->end()) {

        // insert the iterator
        this->pageIterators.push(it);

        // push the page
        this->pages.push_back(page);
      }
    }

    // mark as initialized
    isInitialized = true;
  }

  std::tuple<pdb::TupleSetPtr, std::vector<std::pair<size_t, size_t>>*, std::vector<uint32_t>> getNextTupleSet() override {

    // initialize the source
    if(!isInitialized) {
      initialize();
    }

    //
    std::vector<uint32_t> rightTDI;

    // if we don't have any pages finish
    if (pageIterators.empty()) {
      TupleSetPtr tmp = nullptr;
      return std::make_tuple(tmp, &counts, rightTDI);
    }

    // use the page
    rightMap->repin();
    UseTemporaryAllocationBlock blk{rightMap->getBytes(), rightMap->getSize()};

    // the type we need to hold
    Handle<pdb::Map<decltype(((RHS*) nullptr)->myData), uint32_t>> tidMap = makeObject<pdb::Map<decltype(((RHS*) nullptr)->myData), uint32_t>>();

    // the current TID
    uint32_t curTID = 0;

    // fill up the output
    int count = 0;
    hashColumn.clear();
    counts.clear();
    while (!pageIterators.empty()) {

      // find the hash
      auto hash = pageIterators.top().getHash();

      // just to make the code look nicer
      auto tmp = pageIterators.top();
      pageIterators.pop();

      // grab the current records
      auto currentRecordsPtr = *tmp;
      auto &currentRecords = *currentRecordsPtr;

      // fill up the output
      for (auto i = 0; i < currentRecords.size(); ++i) {

        // if we don't have a tid assign one
        if(tidMap->count(currentRecords[i].myData) == 0) {
          rightTDI.emplace_back(curTID);
          (*tidMap)[currentRecords[i].myData] = curTID++;
        }
        else {

          // insert the TID
          rightTDI.emplace_back((*tidMap)[currentRecords[i].myData]);
        }

        // unpack the record
        unpack(currentRecords[i], count++, 0, columns);
        hashColumn.emplace_back(hash);
      }

      // insert the counts
      if (counts.empty() || counts.back().first != hash) {
        counts.emplace_back(make_pair(hash, 0));
      }

      // set the number of counts
      counts.back().second += currentRecords.size();

      // reinsert the next iterator
      ++tmp;
      if (!tmp.isDone()) {

        // reinsert the iterator
        pageIterators.push(tmp);
      }
    }

    // truncate if we have extra
    eraseEnd<RHS>(count, 0, columns);
    hashColumn.resize((unsigned) count);

    // return the output
    return std::make_tuple(output, &counts, std::move(rightTDI));
  }

protected:

  // this is where we store the map with all keys and TIDs
  PDBPageHandle rightMap;

  // is this source initialized
  bool isInitialized = false;

  // and the tuple set we return
  TupleSetPtr output;

  // tells us which attribute is the key
  int keyAtt{};

  // the attribute order of the records
  std::vector<int> recordAttributes;

  // to setup the output tuple set
  TupleSetSetupMachine myMachine;

  // the page set we are going to be grabbing the pages from
  PDBAbstractPageSetPtr pageSet;

  // the left hand side maps
  std::vector<Handle<JoinMap<RHS>>> maps;

  // the iterators of the map
  std::priority_queue<JoinMapIterator < RHS>, std::vector<JoinMapIterator < RHS>>, JoinIteratorComparator<RHS>> pageIterators;

  // pages that contain lhs side pages
  std::vector<PDBPageHandle> pages;

  // this is the worker we are doing the processing for
  uint64_t workerID = 0;

  // the output columns of the tuple set
  void **columns{};

  // the vector that contains the hash column
  std::vector<size_t> hashColumn;

  // the counts of the same hash
  std::vector<pair<size_t, size_t>> counts;
};

}