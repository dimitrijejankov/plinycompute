#include <cstring>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <random>
#include <gtest/gtest.h>
#include <PDBFeedingPageSet.h>

#include "PDBBufferManagerImpl.h"
#include "PDBPageHandle.h"
#include "PDBSet.h"

namespace pdb {

TEST(FeedingPageSetTest, Test1) {

  const uint64_t numThreads = 16;
  const uint64_t pagesPerThread = 1000;

  // create the buffer manager
  PDBBufferManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64, 30, "metadata", ".");

  // the feeding page set
  auto feedingPageSet = std::make_shared<PDBFeedingPageSet>(numThreads / 2, numThreads / 2);


  std::vector<std::shared_ptr<std::thread>> threads;
  threads.reserve(numThreads);
  for(uint64_t i = 0; i < numThreads; ++i) {

    if(i < numThreads / 2) {

      uint64_t thread = i;

      // feeder thread
      threads.emplace_back(std::make_shared<std::thread>([&, thread]() {

        for(uint64_t p = 0; p < pagesPerThread; p++) {

          // just so we get more concurency with the other thread
          usleep(100);

          // get the page write something to it
          auto page = myMgr.getPage();
          *((uint64_t*) page->getBytes()) = thread * pagesPerThread + p;

          // unpin it
          page->unpin();

          // feed it into the page set
          feedingPageSet->feedPage(page);
        }

        // finish feeding
        feedingPageSet->finishFeeding();

      }));
    }
    else {

      uint64_t thread = i - (numThreads / 2);

      // reader thread
      threads.emplace_back(std::make_shared<std::thread>([&, thread]() {

        // sum up the stuff
        uint64_t sum = 0;

        PDBPageHandle page;
        while((page = feedingPageSet->getNextPage(thread)) != nullptr) {
          sum += *((uint64_t*) page->getBytes());
        }

        // check if we got the right value
        EXPECT_EQ(sum, (pagesPerThread * numThreads / 2) * ((pagesPerThread * numThreads / 2) - 1) / 2);
      }));
    }
  }

  // wait to finish
  for(auto &t : threads){
    t->join();
  }

}

}