
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
TEST(BufferManagerDebugTest, Test1) {

  const int pageSize = 64;

  // create the buffer manager
  pdb::PDBBufferManagerDebugFrontend myMgr("tempDSFSD", pageSize, 2, "metadata", ".");

  // create the three sets
  vector<PDBSetPtr> mySets;
  vector<unsigned> myEnds;
  vector<vector<size_t>> lens;
  for (int i = 0; i < 6; i++) {
    PDBSetPtr set = make_shared<PDBSet>("DB" + to_string(i), "set");
    mySets.push_back(set);
    myEnds.push_back(0);
    lens.emplace_back(vector<size_t>());
  }

  // now, we create a bunch of data and write it to the files, unpinning it
  for (int i = 0; i < 10; i++) {
    PDBPageHandle temp = createRandomPage(myMgr, mySets, myEnds, lens);
    temp->unpin();
  }

  // the buffer
  char buffer[1024];

  // for each set
  for (int i = 0; i < 6; i++) {

    // for each page
    for (int j = 0; j < myEnds[i]; j++) {

      // grab the page
      PDBPageHandle temp = myMgr.getPage(mySets[i], (uint64_t) j);

      // generate the right string
      writeBytes(i, j, (int) lens[i][j], (char *) buffer);

      // check the string
      EXPECT_EQ(strcmp(buffer, (char*) temp->getBytes()), 0);
    }
  }
}

// same as test 12 but with different page size and same processing page size and has tracing
TEST(BufferManagerDebugTest, Test2) {

  // create the buffer manager
  PDBBufferManagerDebugFrontend myMgr("tempDSFSD", 256, 4, "metadata", ".");

  // the page sizes we are testing
  std::vector<size_t> pageSizes {8, 16, 32, 64, 128};

  const int numRequestsPerPage = 100;
  const int numPages = 60;

  // note the number of threads must be less than 8 or equal to 8 or else we can exceed the page size
  const int numThreads = 4;

  // generate the pages
  PDBSetPtr set = make_shared<PDBSet>("DB", "set1");
  for(uint64_t i = 0; i < numPages; ++i) {

    // grab the page
    auto page = myMgr.getPage(set, i);

    // freeze the size
    page->freezeSize(pageSizes[i % 5]);

    for(int t = 0; t < numThreads; ++t) {
      // set the first numThreads bytes to 0
      ((char *) page->getBytes())[t] = 0;
    }

    // mark as dirty
    page->setDirty();
  }

  std::atomic<std::int32_t> sync;
  sync = 0;

  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for(int t = 0; t < numThreads; ++t) {

    threads.emplace_back(std::thread([&](int tmp) {

      int myThraed = tmp;
      int myThreadClamp = ((myThraed + 1) * 100) % 127;

      // generate the page indices
      std::vector<uint64_t> pageIndices;
      for(int i = 0; i < numRequestsPerPage; ++i) {
        for(int j = 0; j < numPages; ++j) {
          pageIndices.emplace_back(j);
        }
      }

      // shuffle the page indices
      auto seed = std::chrono::system_clock::now().time_since_epoch().count();
      shuffle (pageIndices.begin(), pageIndices.end(), std::default_random_engine(seed));

      sync++;
      while (sync != numThreads) {}
      for(auto it : pageIndices) {

        // grab the page
        auto page = myMgr.getPage(set, it);

        // increment the page
        ((char *) page->getBytes())[myThraed] = (char) ((((char *) page->getBytes())[myThraed] + 1) % myThreadClamp);

        // set as dirty
        page->setDirty();
      }
    }, t));
  }

  for(auto &t : threads) {
    t.join();
  }

  for(uint64_t i = 0; i < numPages; ++i) {

    // the page
    auto page = myMgr.getPage(set, i);

    for(int t = 0; t < numThreads; ++t) {

      int myThreadClamp = ((t + 1) * 100) % 127;

      // check them
      EXPECT_EQ(((char*) page->getBytes())[t], (numRequestsPerPage % myThreadClamp));
    }
  }
}

// same as test 12 but with different page size and same processing page size and has tracing
TEST(BufferManagerDebugTest, Test3) {

  // create the buffer manager
  PDBBufferManagerDebugFrontend myMgr("tempDSFSD", 256, 4, "metadata", ".");

  auto page1 = myMgr.getPage();
  auto page2 = myMgr.getPage();
  page1->freezeSize(32);
  page2->freezeSize(32);
  page1->unpin();
  page2->unpin();

  auto page3 = myMgr.getPage();
  auto page4 = myMgr.getPage();
  page3->freezeSize(32);
  page4->freezeSize(32);
  page3->unpin();
  page4->unpin();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}