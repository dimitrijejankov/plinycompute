#pragma once

#include <utility>
#include "JoinAggTupleEmitter.h"
#include "TRABlock.h"

namespace pdb {

class TRALocalJoinEmitter : public JoinAggTupleEmitterInterface {
public:

  struct ThreadInfo {
    std::mutex m;
    std::condition_variable cv;
    bool gotRecords{false};
    std::vector<JoinedRecord> buffer;
  };


  TRALocalJoinEmitter(std::size_t numThreads,
                      PDBSetPageSetPtr input_page_set,
                      PDBRandomAccessPageSetPtr left_page_set,
                      PDBRandomAccessPageSetPtr right_page_set,
                      const pdb::Vector<int32_t> &lhsIndices,
                      const pdb::Vector<int32_t> &rhsIndices,
                      TRAIndexNodePtr index) : inputPageSet(std::move(input_page_set)),
                                                                leftPageSet(std::move(left_page_set)),
                                                                rightPageSet(std::move(right_page_set)),
                                                                numThreads(numThreads),
                                                                threadsWaiting(numThreads),
                                                                lhsIndices(lhsIndices),
                                                                rhsIndices(rhsIndices),
                                                                index(std::move(std::move(index))) {}

  void getRecords(std::vector<JoinedRecord> &putHere,
                  int32_t &lastLHSPage,
                  int32_t &lastRHSPage,
                  int32_t threadID) override {

    // clear the vector
    putHere.clear();

    {

      // lock this and add records
      std::unique_lock<std::mutex> lck(m);

      // reset the value so emitter knows we need to wait
      if(threadsWaiting[threadID].buffer.empty()) {
        threadsWaiting[threadID].gotRecords = false;
      }

      // wait till we get some records
      while (!threadsWaiting[threadID].gotRecords && !hasEnded) {
        threadsWaiting[threadID].cv.wait(lck);
      }

      // swap the threads
      std::swap(putHere, threadsWaiting[threadID].buffer);

      // the LHS pages are added
      lastLHSPage = leftPageSet->getNumPages();

      // the RHS pages are constant
      lastRHSPage = rightPageSet->getNumPages();
    }
  }

  void run() {

    std::vector<std::pair<int32_t, int32_t>> rhsMatch;

    PDBPageHandle page;
    while((page = inputPageSet->getNextPage(0))) {

      // get the vector from the page
      auto &vec = *(((Record<Vector<Handle<TRABlock>>> *) page->getBytes())->getRootObject());

      // generate the index
      for(int i = 0; i < vec.size(); ++i) {

        // store the page in the indexed page set
        auto loc = leftPageSet->pushPage(page);

        // lhs record position
        auto lhsPos = std::make_pair<uint32_t, uint32_t>( i, loc );

        // if this is too slow, we can optimize it
        std::vector<int32_t> rhsMatcher(vec[i]->metaData->indices.size(), -1);

        // figure out the matching pattern
        getRHSMatcher(vec[i]->metaData->indices, rhsMatcher);

        // query the index
        rhsMatch.clear();
        index->get(rhsMatch, rhsMatcher);

        for(auto &rm : rhsMatch) {
          threadsWaiting[currentThread].buffer.emplace_back(loc, i, rm.first, rm.second);
        }
      }
    }
  }

  // needs to accept a vector with -1 at least on the first usage...
  // will move this
  void getRHSMatcher(const pdb::Vector<uint32_t> &lhs_index, std::vector<int32_t> &rhsMatch) {

    for(int i = 0; i < lhsIndices.size(); ++i) {
      rhsMatch[rhsIndices[i]] = lhs_index[lhsIndices[i]];
    }
  }

  int32_t currentThread = 0;

  // did we end?
  bool hasEnded = false;

  // the number of threads
  std::size_t numThreads;

  // the lhs indices
  const pdb::Vector<int32_t> &lhsIndices;

  // the rhs indices
  const pdb::Vector<int32_t> &rhsIndices;

  // this locks this
  std::mutex m;

  // all the threads that are waiting for records to join
  std::vector<ThreadInfo> threadsWaiting;

  // the the input page set
  pdb::PDBSetPageSetPtr inputPageSet;

  // the emmitter will put set pageser here
  pdb::PDBRandomAccessPageSetPtr leftPageSet;

  // get the rhs page set
  pdb::PDBRandomAccessPageSetPtr rightPageSet;

  // the index
  TRAIndexNodePtr index;
};

}