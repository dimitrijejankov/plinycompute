#include <cstring>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <random>
#include <gtest/gtest.h>
#include <thread>

#include "PDBStorageManagerImpl.h"
#include "PDBPageHandle.h"
#include "PDBSet.h"

using namespace std;
using namespace pdb;

int main(int argc, char **argv) {

  // create the storage manager
  PDBStorageManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64, 5, "metadata", ".");

  // the page sizes we are testing
  std::vector<size_t> pageSizes {8, 16, 32};

  const int numRequestsPerPage = 2000;
  const int numPages = 20;
  const int numThreads = 2;

  // generate the pages
  PDBSetPtr set = make_shared<PDBSet>("set1", "DB");
  for(uint64_t i = 0; i < numPages; ++i) {

    // grab the page
    auto page = myMgr.getPage(set, i);

    // freeze the size
    page->freezeSize(pageSizes[i % 3]);

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
      if (((int*) page->getBytes())[t] != numRequestsPerPage) {
        std::cout << "why " << ((int*) page->getBytes())[t] << std::endl;
      }
    }
  }

  return 0;
}
