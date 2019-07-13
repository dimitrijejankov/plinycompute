
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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}