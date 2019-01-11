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
          if(bytes[j] != static_cast<char>((j + offset + tmp) % 128)) {
            std::cout << "why?" << std::endl;
          }
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


  return 0;
}
