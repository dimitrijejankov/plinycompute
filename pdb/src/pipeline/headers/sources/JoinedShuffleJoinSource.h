#include <utility>

#pragma once

#include <ComputeSource.h>
#include <JoinPairArray.h>
#include <utility>

namespace pdb {

template<typename LHS>
class JoinedShuffleJoinSource : public ComputeSource {

protected:

  // and the tuple set we return
  TupleSetPtr output;

  // the size of the last output
  uint64_t outputSize = 0;

  // the buffer we use if we generate to many records
  TupleSetPtr buffer;

  // number of records left in the buffer
  uint64_t bufferSize = 0;

  // to setup the output tuple set
  TupleSetSetupMachine rhsMachine;

  // the source we are going to grab the rhs tuples from
  RHSShuffleJoinSourceBasePtr rhsSource;

  // the attribute order of the records
  std::vector<int> lhsRecordOrder;

  // the left hand side maps
  std::vector<Handle<JoinMap<LHS>>> lhsMaps;

  // the iterators of the map
  std::priority_queue<JoinMapIterator<LHS>, std::vector<JoinMapIterator<LHS>>, JoinIteratorComparator<LHS>> lhsIterators;

  // pages that contain lhs side pages
  std::vector<PDBPageHandle> lhsPages;

  // this is the worker we are doing the processing for
  uint64_t workerID = 0;

  // the output columns of the tuple set
  void **lhsColumns{};

  // the offset where the right input is going to be
  int offset{};

  // the list of counts for matches of each of the rhs tuples. Basically if the count[3] = 99 the fourth tuple in the rhs tupleset will be repeated 99 times
  std::vector<uint32_t> counts;

  // list of iterators that are
  std::vector<JoinMapIterator<LHS>> currIterators;

  // is this source initialized
  bool isInitialized = false;

  // the left input page set
  PDBAbstractPageSetPtr lhsInputPageSet;

public:

  JoinedShuffleJoinSource(TupleSpec &inputSchemaRHS,
                          TupleSpec &hashSchemaRHS,
                          TupleSpec &recordSchemaRHS,
                          PDBAbstractPageSetPtr lhsInputPageSet,
                          const std::vector<int> &lhsRecordOrder,
                          RHSShuffleJoinSourceBasePtr &rhsSource,
                          bool needToSwapLHSAndRhs,
                          uint64_t workerID) : lhsRecordOrder(lhsRecordOrder),
                                               rhsMachine(inputSchemaRHS, recordSchemaRHS),
                                               lhsInputPageSet(std::move(lhsInputPageSet)),
                                               rhsSource(rhsSource),
                                               workerID(workerID) {

    // set up the output tuple and buffer
    output = std::make_shared<TupleSet>();
    buffer = std::make_shared<TupleSet>();
    lhsColumns = new void *[lhsRecordOrder.size()];

    // were the RHS and the LHS side swapped?
    if (!needToSwapLHSAndRhs) {

      // the right input will be put on offset-th column of the tuple set
      offset = (int) lhsRecordOrder.size();

      // the left input will be put at position 0
      createCols<LHS>(lhsColumns, *output, 0, 0, lhsRecordOrder);
    } else {

      // the right input will be put at the begining of the tuple set
      offset = 0;

      // the left input will be put at the recordOrder.size()-th column
      createCols<LHS>(lhsColumns, *output, (int) recordSchemaRHS.getAtts().size(), 0, lhsRecordOrder);
    }
  }

  ~JoinedShuffleJoinSource() override {

    // unpin the pages
    for_each (lhsPages.begin(), lhsPages.end(), [&](PDBPageHandle &page) { page->unpin(); });

    // delete the columns
    delete[] lhsColumns;
  }

  TupleSetPtr getNextTupleSet(const PDBTupleSetSizePolicy &policy) override {


    // if it is not initialized initialize it
    if(!isInitialized) {

      PDBPageHandle page;
      while((page = lhsInputPageSet->getNextPage(workerID)) != nullptr) {

        // pin the page
        page->repin();

        // we grab the vector of hash maps
        Handle<Vector<Handle<JoinMap<LHS>>>> returnVal = ((Record<Vector<Handle<JoinMap<LHS>>>> *) (page->getBytes()))->getRootObject();

        // next we grab the join map we need
        lhsMaps.push_back((*returnVal)[workerID]);

        if(lhsMaps.back()->size() != 0) {
          // insert the iterator
          lhsIterators.push(lhsMaps.back()->begin());

          // push the page
          lhsPages.push_back(page);
        }
      }

      // mark it as  true
      isInitialized = true;
    }


    /**
     * 0. In case of failure we need to reprocess the input, copy the current stuff into the buffer
     */

    // did we manage to process the input, if not move the records into the buffer
    if(!policy.inputWasProcessed()) {

      // merge the previous output into the buffer since we failed
      TupleSet::merge(*buffer, *output);
      bufferSize += outputSize;
      outputSize = 0;
    }

    /**
     * 1. We need to check if the buffer has something, if it does we need to process it.
     */

    if(bufferSize != 0) {

      // figure out how many records we want to grab (not more than the buffer size)
      outputSize = std::min<uint64_t>(bufferSize, policy.getChunksSize());

      // this one is a bit tricky basically it moves a outputSize number of records into the buffer
      // by splitting on the outputSize-th position from the back of the buffer, since getting rows from the back is
      // faster
      TupleSet::split(*buffer, *output, bufferSize - outputSize);
      bufferSize -= outputSize;

      // return the output
      return output;
    }

    // get the rhs tuple
    auto rhsTuple = rhsSource->getNextTupleSet();
    if(rhsTuple.first == nullptr) {
      return nullptr;
    }

    // clear the counts from the previous call
    counts.clear();

    // go through the hashes
    outputSize = 0;
    for (auto &currHash : *rhsTuple.second) {

      // clear the iterators from the previous iteration
      currIterators.clear();

      // get the hash and count
      auto &rhsHash = currHash.first;
      auto &rhsCount = currHash.second;

      // at the beginning we assume that there were not matches for the rhs on the lhs side
      for(int i = 0; i < rhsCount; ++i) {
        counts.emplace_back(0);
      }

      /// 1. Figure out if there is an lhs hash equal to the rhs hash

      // if the left hand side does not have any iterators just skip
      if(lhsIterators.empty()) {
        continue;
      }

      // check if the lhs hash is above the rhs hash if it is there is nothing to join, go to the next records
      if(lhsIterators.top().getHash() > rhsHash) {
        continue;
      }

      // iterate till we either find a lhs hash that is equal or greater to the rhs one, or finish
      JoinMapIterator<LHS> curIterator;
      size_t lhsHash{};
      do {

        // if we are out of iterators this means we are done, break out of this loop
        if(lhsIterators.empty()) {
          break;
        }

        // grab the current lhs hash
        curIterator = lhsIterators.top();
        lhsHash = curIterator.getHash();

        // check if the lhs hash is above the rhs hash if it is there is nothing to join break out of this loop
        if(lhsHash > rhsHash) {
          break;
        }

        // move the iterator, and reinsert it into the queue since we either have a match or we are below the hash
        lhsIterators.pop();
        auto nextIterator = curIterator + 1;
        if(!nextIterator.isDone()) {
          lhsIterators.push(nextIterator);
        }

      } while (lhsHash < rhsHash);

      // if we don't have a match skip this
      if(lhsHash > rhsHash) {
        continue;
      }

      // store the current iterator, since at this point it lhs and rhs have the same hash
      currIterators.emplace_back(curIterator);

      /// 1. Figure out if there are more iterators in the lhs that mach our rhs iterator

      do {

        // if we don't have stuff one iterator is enough
        if(lhsIterators.empty()) {
          break;
        }

        // check if we still have stuff to join
        if(lhsIterators.top().getHash() == rhsHash) {

          // update the current iterator
          curIterator = lhsIterators.top();
          lhsIterators.pop();

          // store the current iterator
          currIterators.emplace_back(curIterator);

          // move the iterator
          auto nextIterator = curIterator + 1;
          if(!nextIterator.isDone()) {
            lhsIterators.push(curIterator + 1);
          }

          continue;
        }

        // do we sill need to do stuff?
      } while(lhsIterators.top().getHash() == rhsHash);

      /// 2. Now for every rhs record we need to replicate every lhs record we could find

      for(auto i = 0; i < rhsCount; ++i) {

        // go through each iterator that has stuff in the lhs
        for(auto &iterator : currIterators) {
          auto it = *iterator;
          auto &records = *it;

          // update the counts
          counts[counts.size() - rhsCount + i] += records.size();

          // emit lhs records in this iterator
          for (int which = 0; which < records.size(); which++) {

            // do the unpack
            unpack(records[which], outputSize++, 0, lhsColumns);
          }
        }
      }
    }

    // truncate if we have extra
    eraseEnd<LHS>(outputSize, 0, lhsColumns);

    /// 3. Finally, we need to replicate the rhs records

    rhsMachine.replicate(rhsTuple.first, output, counts, offset);

    /// 4. if we have more than the chunks size buffer the extra rows

    if(outputSize > policy.getChunksSize()) {
      TupleSet::split(*output, *buffer, policy.getChunksSize());
      bufferSize = outputSize - policy.getChunksSize();
      outputSize = policy.getChunksSize();
    }

    return output;
  }

};

}
