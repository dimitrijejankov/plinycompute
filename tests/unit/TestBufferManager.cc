
#ifndef CATALOG_UNIT_H
#define CATALOG_UNIT_H

#include <cstring>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <random>
#include <gtest/gtest.h>

#include "PDBStorageManagerImpl.h"
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

PDBPageHandle createRandomPage(PDBStorageManagerImpl &myMgr, vector<PDBSetPtr> &mySets, vector<unsigned> &myEnds, vector<vector<size_t>> &lens) {

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
PDBPageHandle createRandomTempPage(PDBStorageManagerImpl &myMgr, vector<size_t> &lengths) {

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

TEST(BufferManagerTest, Test1) {

  // create the storage manager
  PDBStorageManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64, 16, "metadata", ".");

  // grab two pages
  PDBPageHandle page1 = myMgr.getPage();
  PDBPageHandle page2 = myMgr.getPage();

  // write 64 bytes to page 2
  char *bytes = (char *) page1->getBytes();
  memset(bytes, 'A', 64);
  bytes[63] = 0;

  // write 32 bytes to page 1
  bytes = (char *) page2->getBytes();
  memset(bytes, 'V', 32);
  bytes[31] = 0;

  // unpin page 1
  page1->unpin();

  // freeze the size to 32 and unpin it
  page2->freezeSize(32);
  page2->unpin();

  // just grab some random pages
  for (int i = 0; i < 32; i++) {
    PDBPageHandle page3 = myMgr.getPage();
    PDBPageHandle page4 = myMgr.getPage();
    PDBPageHandle page5 = myMgr.getPage();
  }

  // repin page 1 and check
  page1->repin();
  bytes = (char *) page1->getBytes();
  EXPECT_EQ(memcmp("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\0", bytes, 64), 0);

  // repin page 2 and check
  page2->repin();
  bytes = (char *) page2->getBytes();
  EXPECT_EQ(memcmp("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV\0", bytes, 32), 0);
}

TEST(BufferManagerTest, Test2) {
  PDBStorageManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64, 16, "metadata", ".");
  PDBSetPtr set1 = make_shared<PDBSet>("set1", "DB");
  PDBSetPtr set2 = make_shared<PDBSet>("set2", "DB");
  PDBPageHandle page1 = myMgr.getPage(set1, 0);
  PDBPageHandle page2 = myMgr.getPage(set2, 0);
  char *bytes = (char *) page1->getBytes();
  memset(bytes, 'A', 64);
  bytes[63] = 0;
  bytes = (char *) page2->getBytes();
  memset(bytes, 'V', 32);
  bytes[31] = 0;
  page1->unpin();
  page2->freezeSize(32);
  page2->unpin();
  for (int i = 0; i < 32; i++) {
    PDBPageHandle page3 = myMgr.getPage(set1, i + 1);
    PDBPageHandle page4 = myMgr.getPage(set1, i + 31);
    PDBPageHandle page5 = myMgr.getPage(set2, i + 1);
    bytes = (char *) page5->getBytes();
    memset(bytes, 'V', 32);
    if (i % 3 == 0) {
      bytes[31] = 0;
      page5->freezeSize(32);
    } else {
      bytes[15] = 0;
      page5->freezeSize(16);
    }
  }
  page1->repin();
  bytes = (char *) page1->getBytes();
  EXPECT_EQ(memcmp("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\0", bytes, 64), 0);

  page2->repin();
  bytes = (char *) page2->getBytes();
  EXPECT_EQ(memcmp("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV\0", bytes, 32), 0);
}

TEST(BufferManagerTest, Test3) {
  PDBStorageManagerImpl myMgr;
  myMgr.initialize("metadata");
  PDBSetPtr set1 = make_shared<PDBSet>("set1", "DB");
  PDBSetPtr set2 = make_shared<PDBSet>("set2", "DB");
  PDBPageHandle page1 = myMgr.getPage(set1, 0);
  PDBPageHandle page2 = myMgr.getPage(set2, 0);

  char *bytes = (char *) page1->getBytes();
  EXPECT_EQ(memcmp("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\0", bytes, 64), 0);

  bytes = (char *) page2->getBytes();
  EXPECT_EQ(memcmp("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV\0", bytes, 32), 0);
}

TEST(BufferManagerTest, Test4) {

  // create a buffer manager
  PDBStorageManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64, 16, "metadata", ".");

  // create the three sets
  vector<PDBSetPtr> mySets;
  vector<unsigned> myEnds;
  vector<vector<size_t>> lens;
  for (int i = 0; i < 6; i++) {
    PDBSetPtr set = make_shared<PDBSet>("set" + to_string(i), "DB");
    mySets.push_back(set);
    myEnds.push_back(0);
    lens.emplace_back(vector<size_t>());
  }

  // now, we create a bunch of data and write it to the files, unpinning it
  for (int i = 0; i < 1000; i++) {
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

TEST(BufferManagerTest, Test5) {
  PDBStorageManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64, 16, "metadata", ".");

  // create the three sets
  vector<PDBSetPtr> mySets;
  vector<unsigned> myEnds;
  vector<vector<size_t>> lens;
  for (int i = 0; i < 6; i++) {
    PDBSetPtr set = make_shared<PDBSet>("set" + to_string(i), "DB");
    mySets.push_back(set);
    myEnds.push_back(0);
    lens.emplace_back(vector<size_t>());
  }

  // first, we create a bunch of data and write it to the files
  vector<PDBPageHandle> myPinnedPages;
  for (int i = 0; i < 10; i++) {
    myPinnedPages.push_back(createRandomPage(myMgr, mySets, myEnds, lens));
  }

  // now, we create a bunch of data and write it to the files, unpinning it
  for (int i = 0; i < 1000; i++) {
    PDBPageHandle temp = createRandomPage(myMgr, mySets, myEnds, lens);
    temp->unpin();
  }

  // now, unpin the temp pages
  for (auto &a : myPinnedPages) {
    a->unpin();
  }

  // next, we create a bunch of temporary data
  vector<PDBPageHandle> myTempPages;
  vector<size_t> tmpPageLengths;
  for (int i = 0; i < 13; i++) {
    myTempPages.push_back(createRandomTempPage(myMgr, tmpPageLengths));
  }

  // next, we create more data and write it to the files
  // now, we create a bunch of data and write it to the files, unpinning it
  for (int i = 0; i < 1000; i++) {
    PDBPageHandle temp = createRandomPage(myMgr, mySets, myEnds, lens);
    temp->unpin();
  }

  // now, unpin the temporary data
  for (auto &a : myTempPages) {
    a->unpin();
  }

  // get rid of the refs to the pinned pages
  myPinnedPages.clear();

  // the buffer
  char buffer[1024];

  // now, check the files
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < myEnds[i]; j++) {

      PDBPageHandle temp = myMgr.getPage(mySets[i], (uint64_t) j);

      // generate the right string
      writeBytes(i, j, (int) lens[i][j], (char *) buffer);

      // check the string
      EXPECT_EQ(strcmp(buffer, (char*) temp->getBytes()), 0);
    }
  }

  // and, check the temp pages
  counter = 0;
  for (auto &a : myTempPages) {

    // repin the page
    a->repin();

    // write to buffer
    writeBytes(-1, counter, tmpPageLengths[counter], (char *) buffer);

    // inc the counter
    counter++;

    // check the string
    EXPECT_EQ(strcmp(buffer, (char*) a->getBytes()), 0);
  }

  myTempPages.clear();
}

TEST(BufferManagerTest, Test6) {

  {
    PDBStorageManagerImpl myMgr;
    myMgr.initialize("tempDSFSD", 64, 16, "metadata", ".");
    PDBSetPtr set1 = make_shared<PDBSet>("set1", "DB");
    PDBSetPtr set2 = make_shared<PDBSet>("set2", "DB");

    char buffer[128];

    for(int j = 0; j < 100; ++j) {

      PDBPageHandle page1 = myMgr.getPage(set1, j);
      PDBPageHandle page2 = myMgr.getPage(set2, j);

      char *bytes = (char *) page1->getBytes();
      memset(bytes, 'A' + (char) (j / 12), 64);
      bytes[63] = 0;
      bytes = (char *) page2->getBytes();
      memset(bytes, 'B' + (char) (j / 12), 32);
      bytes[31] = 0;
      page1->unpin();
      page2->freezeSize(32);
      page2->unpin();

      // just grab some random pages
      for (int i = 0; i < 32; i++) {
        PDBPageHandle page3 = myMgr.getPage();
        PDBPageHandle page4 = myMgr.getPage();
        PDBPageHandle page5 = myMgr.getPage();
      }

      page1->repin();
      bytes = (char *) page1->getBytes();
      memset((char*) buffer, 'A' + (char) (j / 12), 64);
      buffer[63] = 0;
      EXPECT_EQ(memcmp(buffer, bytes, 64), 0);

      page2->repin();
      bytes = (char *) page2->getBytes();
      memset((char*) buffer, 'B' + (char) (j / 12), 32);
      buffer[31] = 0;
      EXPECT_EQ(memcmp(buffer, bytes, 32), 0);
    }
  }

  {
    PDBStorageManagerImpl myMgr;
    myMgr.initialize("metadata");

    char buffer[64];

    for(int j = 0; j < 100; ++j) {

      PDBSetPtr set1 = make_shared<PDBSet>("set1", "DB");
      PDBSetPtr set2 = make_shared<PDBSet>("set2", "DB");
      PDBPageHandle page1 = myMgr.getPage(set1, j);
      PDBPageHandle page2 = myMgr.getPage(set2, j);

      memset((char*) buffer, 'A' + (char) (j / 12), 64);
      buffer[63] = 0;
      page1->repin();
      char *bytes = (char *) page1->getBytes();
      EXPECT_EQ(memcmp(buffer, bytes, 64), 0);

      memset((char*) buffer, 'B' + (char) (j / 12), 32);
      buffer[31] = 0;
      page2->repin();
      bytes = (char *) page2->getBytes();
      EXPECT_EQ(memcmp(buffer, bytes, 32), 0);
    }
  }
}

// this tests tests concurrent set pages
TEST(BufferManagerTest, Test7) {

  // create the storage manager
  PDBStorageManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64, 5, "metadata", ".");

  const int numRequestsPerPage = 2000;
  const int numPages = 20;
  const int numThreads = 4;

  // generate the pages
  PDBSetPtr set = make_shared<PDBSet>("set1", "DB");
  for(uint64_t i = 0; i < numPages; ++i) {

    // grab the page
    auto page = myMgr.getPage(set, i);

    for(int t = 0; t < numThreads; ++t) {
      // set the first 4 bytes to 0
      ((int *) page->getBytes())[t] = 0;
    }

    // mark as dirty
    page->setDirty();
  }

  atomic_int32_t sync;
  sync = 0;

  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for(int t = 0; t < numThreads; ++t) {

    threads.emplace_back(std::thread([&](int tmp) {

      int myThraed = tmp;

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
        ((int*) page->getBytes())[myThraed]++;

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

      // check them
      EXPECT_EQ (((int*) page->getBytes())[t], numRequestsPerPage);
    }
  }
}

// this test tests concurrent anonymous pages
TEST(BufferManagerTest, Test8) {

  // create the storage manager
  PDBStorageManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64, 10, "metadata", ".");

  // parameters
  const int numPages = 1000;
  const int numThreads = 4;

  // used to sync
  atomic_int32_t sync;
  sync = 0;

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

        // grab the page
        auto page = myMgr.getPage();

        // grab the page and fill it in
        char* bytes = (char*) page->getBytes();
        for(char j = 0; j < 64; ++j) {
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
      for(auto &page : pageHandles) {

        // repin the page
        page->repin();

        // grab the page and fill it in
        char* bytes = (char*) page->getBytes();
        for(char j = 0; j < 64; ++j) {
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

#endif
