#pragma once

#include "RHSKeyJoinSourceBase.h"

#include <utility>


namespace pdb {

template<typename LHS>
class JoinedKeySource : public ComputeSource {
public:

  JoinedKeySource(TupleSpec &inputSchemaRHS,
                  TupleSpec &hashSchemaRHS,
                  TupleSpec &recordSchemaRHS,
                  const PDBAbstractPageSetPtr &lhsInputPageSet,
                  const std::vector<int> &lhsRecordOrder,
                  RHSKeyJoinSourceBasePtr &rhsSource,
                  bool needToSwapLHSAndRhs,
                  PDBPageHandle leftMap) : leftTDI(new std::vector<uint32_t>),
                                           rightTDI(new std::vector<uint32_t>),
                                           lhsRecordOrder(lhsRecordOrder),
                                           rhsMachine(inputSchemaRHS, recordSchemaRHS),
                                           rhsSource(rhsSource),
                                           leftMap(std::move(leftMap)) {

    PDBPageHandle page;
    while((page = lhsInputPageSet->getNextPage(0)) != nullptr) {

      // pin the page
      page->repin();

      // we grab the vector of hash maps
      Handle<JoinMap<LHS>> returnVal = ((Record<JoinMap<LHS>> *) (page->getBytes()))->getRootObject();

      // next we grab the join map we need
      this->lhsMaps.push_back(returnVal);

      if(this->lhsMaps.back()->size() != 0) {
        // insert the iterator
        this->lhsIterators.push(this->lhsMaps.back()->begin());

        // push the page
        this->lhsPages.push_back(page);
      }
    }

    // set up the output tuple
    this->output = std::make_shared<TupleSet>();
    this->lhsColumns = new void *[lhsRecordOrder.size()];

    // were the RHS and the LHS side swapped?
    if (!needToSwapLHSAndRhs) {

      // the right input will be put on offset-th column of the tuple set
      this->offset = (int) lhsRecordOrder.size();

      // the left input will be put at position 0
      createCols<LHS>(this->lhsColumns, *this->output, 0, 0, lhsRecordOrder);
    } else {

      // the right input will be put at the begining of the tuple set
      this->offset = 0;

      // the left input will be put at the recordOrder.size()-th column
      createCols<LHS>(this->lhsColumns, *this->output, (int) recordSchemaRHS.getAtts().size(), 0, lhsRecordOrder);
    }

    // add the columns for the TIDs
    this->output->addColumn(this->offset + 1, leftTDI, true);
    this->output->addColumn(this->offset + 2, rightTDI, true);
  }

  ~JoinedKeySource() override {
    delete[] lhsColumns;
  }

  TupleSetPtr getNextTupleSet(const PDBTupleSetSizePolicy &policy) override {

    // get the rhs tuple
    auto rhsTuple = rhsSource->getNextTupleSet();
    if(std::get<0>(rhsTuple) == nullptr) {
      return nullptr;
    }

    // use the page
    leftMap->repin();
    UseTemporaryAllocationBlock blk{leftMap->getBytes(), leftMap->getSize()};

    // the type we need to hold
    Handle<pdb::Map<decltype(((LHS*) nullptr)->myData), uint32_t>> tidMap = makeObject<pdb::Map<decltype(((LHS*) nullptr)->myData), uint32_t>>();

    // clear the counts from the previous call
    counts.clear();

    // the current TID
    uint32_t curTID = 0;

    // clear the column of TIDs
    leftTDI->clear();
    rightTDI->clear();

    // go through the hashes
    int overallCounter = 0;
    for (auto &currHash : *std::get<1>(rhsTuple)) {

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

            // if we don't have a tid assign one
            if(tidMap->count(records[which].myData) == 0) {
              leftTDI->emplace_back(curTID);
              (*tidMap)[records[which].myData] = curTID++;
            }
            else {

              // insert the TID
              leftTDI->emplace_back((*tidMap)[records[which].myData]);
            }

            // do the unpack
            unpack(records[which], overallCounter++, 0, lhsColumns);
          }
        }
      }
    }

    // truncate if we have extra
    eraseEnd<LHS>(overallCounter, 0, lhsColumns);

    /// 3. We need to replicate the rhs records

    rhsMachine.replicate(std::get<0>(rhsTuple), output, counts, offset);

    /// 4. set the rhs TIDs

    // replace the rhs TIDs for the lhs
    int idx = 0;
    for(const auto &c : counts) {

      // replicate it
      for(int jdx = 0; jdx < c; jdx++) {
        rightTDI->emplace_back(std::get<2>(rhsTuple)[idx]);
      }

      // increment the idx
      idx++;
    }

    return output;
  }

private:

  // and the tuple set we return
  TupleSetPtr output;

  // the left and right TDI
  std::vector<uint32_t> *leftTDI;
  std::vector<uint32_t> *rightTDI;

  // to setup the output tuple set
  TupleSetSetupMachine rhsMachine;

  // the source we are going to grab the rhs tuples from
  RHSKeyJoinSourceBasePtr rhsSource;

  // the attribute order of the records
  std::vector<int> lhsRecordOrder;

  // the left hand side maps
  std::vector<Handle<JoinMap<LHS>>> lhsMaps;

  // the iterators of the map
  std::priority_queue<JoinMapIterator<LHS>, std::vector<JoinMapIterator<LHS>>, JoinIteratorComparator<LHS>> lhsIterators;

  // pages that contain lhs side pages
  std::vector<PDBPageHandle> lhsPages;

  // the output columns of the tuple set
  void **lhsColumns{};

  // the offset where the right input is going to be
  int offset{};

  // the list of counts for matches of each of the rhs tuples. Basically if the count[3] = 99 the fourth tuple in the rhs tupleset will be repeated 99 times
  std::vector<uint32_t> counts;

  // list of iterators that are
  std::vector<JoinMapIterator<LHS>> currIterators;

  // the left map
  PDBPageHandle leftMap;
};

}
