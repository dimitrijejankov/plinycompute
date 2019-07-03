
#include <cstring>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <random>
#include <gtest/gtest.h>

#include "PDBBufferManagerDebugFrontend.h"
#include "PDBPageHandle.h"
#include "PDBSet.h"

using namespace std;
using namespace pdb;

void writeBytes(int fileName, int pageNum, int pageSize, char *toMe) {

  char foo[1000];
  int num = 0;
  while (num < 900)
    num += sprintf(foo + num, "F: %d, P: %d ", fileName, pageNum);
  memcpy(toMe, foo, pageSize);
  sprintf(toMe + pageSize - 5, "END#");
}

PDBPageHandle createRandomPage(PDBBufferManagerDebugFrontend &myMgr, vector<PDBSetPtr> &mySets, vector<unsigned> &myEnds, vector<vector<size_t>> &lens) {

  // choose a set
  auto whichSet = lrand48() % mySets.size();

  // choose a length
  size_t len = 16;
  for (; (lrand48() % 3 != 0) && (len < 64); len *= 2);

  // store the random len
  lens[whichSet].push_back(len);

  PDBPageHandle returnVal = myMgr.getPage(mySets[whichSet], myEnds[whichSet]);
  writeBytes(whichSet, myEnds[whichSet], len, (char *) returnVal->getBytes());
  myEnds[whichSet]++;
  returnVal->freezeSize(len);
  return returnVal;
}

static int counter = 0;
PDBPageHandle createRandomTempPage(PDBBufferManagerDebugFrontend &myMgr, vector<size_t> &lengths) {

  // choose a length
  size_t len = 16;
  for (; (lrand48() % 3 != 0) && (len < 64); len *= 2);

  // store the length
  lengths.push_back(len);

  PDBPageHandle returnVal = myMgr.getPage();
  writeBytes(-1, counter, len, (char *) returnVal->getBytes());
  counter++;
  returnVal->freezeSize(len);
  return returnVal;
}


// tests anonymous pages of different sizes 8, 16, 32 when the largest page size is 64
TEST(BufferManagerTest, Test1) {

  const int pageSize = 64;

  // create the buffer manager
  pdb::PDBBufferManagerDebugFrontend myMgr("tempDSFSD", pageSize, 10, "metadata", ".");

  // parameters
  const int numPages = 1000;
  const int numThreads = 4;

  // used to sync
  std::atomic<std::int32_t> sync;
  sync = 0;

  // the page sizes we are testing
  std::vector<size_t> pageSizes {8, 16, 32};

  // run multiple threads
  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for(int t = 0; t < numThreads; ++t) {

    threads.emplace_back(std::thread([&](int tmp) {

      // sync the threads to make sure there is more overlapping
      sync++;
      while (sync != numThreads) {}

      int offset = 0;

      std::vector<PDBPageHandle> pageHandles;

      // grab anon pages
      for(int i = 0; i < numPages; ++i) {

        // use a different page size
        size_t pageSize = pageSizes[i % 3];

        // grab the page
        auto page = myMgr.getPage(pageSize);

        // grab the page and fill it in
        char* bytes = (char*) page->getBytes();
        for(char j = 0; j < pageSize; ++j) {
          bytes[j] = static_cast<char>((j + offset + tmp) % 128);
        }

        // store page
        pageHandles.push_back(page);

        // unpin the page
        page->unpin();

        // increment the offset
        offset++;
      }

      // sync the threads to make sure there is more overlapping
      sync++;
      while (sync != 2 * numThreads) {}

      offset = 0;
      for(int i = 0; i < numPages; ++i) {

        // use a different page size
        size_t pageSize = pageSizes[i % 3];

        // grab the page
        auto page = pageHandles[i];

        // repin the page
        page->repin();

        // grab the page and fill it in
        char* bytes = (char*) page->getBytes();
        for(char j = 0; j < pageSize; ++j) {
          EXPECT_EQ(bytes[j], static_cast<char>((j + offset + tmp) % 128));
        }

        // unpin the page
        page->unpin();

        // increment the offset
        offset++;
      }

      // remove all the page handles
      pageHandles.clear();

    }, t));
  }

  for(auto &t : threads) {
    t.join();
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}