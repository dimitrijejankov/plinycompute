#pragma once

#include <utility>
#include <RHSShuffleJoinSourceBase.h>
#include <ComputeSource.h>
#include <TupleSpec.h>
#include <TupleSetMachine.h>
#include <JoinTuple.h>
#include <queue>
#include <PDBAbstractPageSet.h>

namespace pdb {

template<typename RHS>
class RHSShuffleJoinSource : public RHSShuffleJoinSourceBase {
 private:

  // and the tuple set we return
  TupleSetPtr output;

  // tells us which attribute is the key
  int keyAtt;

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

  // the number of tuples in the tuple set
  uint64_t chunkSize = 0;

  // this is the worker we are doing the processing for
  uint64_t workerID = 0;

  // the output columns of the tuple set
  void **columns;

  // the vector that contains the hash column
  std::vector<size_t> hashColumn;

  // the counts of the same hash
  std::vector<pair<size_t, size_t>> counts;

 public:

  RHSShuffleJoinSource() = default;

  RHSShuffleJoinSource(TupleSpec &inputSchema,
                       TupleSpec &hashSchema,
                       TupleSpec &recordSchema,
                       std::vector<int> &recordOrder,
                       PDBAbstractPageSetPtr rightInputPageSet,
                       uint64_t chunkSize,
                       uint64_t workerID) : myMachine(inputSchema), pageSet(std::move(rightInputPageSet)), chunkSize(chunkSize), workerID(workerID) {

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

    PDBPageHandle page;
    while ((page = pageSet->getNextPage(workerID)) != nullptr) {

      // pin the page
      page->repin();

      // we grab the vector of hash maps
      Handle<Vector<Handle<JoinMap<RHS>>>> returnVal = ((Record<Vector<Handle<JoinMap<RHS>>>> *) (page->getBytes()))->getRootObject();

      // next we grab the join map we need
      maps.push_back((*returnVal)[workerID]);

      // if the map has stuff add it to the queue
      auto it = maps.back()->begin();
      if (it != maps.back()->end()) {

        // insert the iterator
        pageIterators.push(it);

        // push the page
        pages.push_back(page);
      }
    }
  }

  ~RHSShuffleJoinSource() override {
    // unpin the pages
    for_each (pages.begin(), pages.end(), [&](PDBPageHandle &page) { page->unpin(); });

    // delete the columns
    delete[] columns;
  }

  std::pair<TupleSetPtr, std::vector<pair<size_t, size_t>>*> getNextTupleSet() override {

    // if we don't have any pages finish
    if (pageIterators.empty()) {
      TupleSetPtr tmp = nullptr;
      return std::make_pair(tmp, &counts);
    }

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

      // did we fill up the tuple set
      if (!pageIterators.empty() && count > chunkSize && pageIterators.top().getHash() != hash) {
        break;
      }
    }

    // truncate if we have extra
    eraseEnd<RHS>(count, 0, columns);
    hashColumn.resize((unsigned) count);

    // return the output
    return std::make_pair(output, &counts);
  }

};

}
