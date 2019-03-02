//
// Created by dimitrije on 1/21/19.
//
#include <cstring>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <random>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <PDBBufferManagerFrontEnd.h>
#include <GenericWork.h>

namespace pdb {

/**
 * This is the mock communicator we provide to the request handlers
 */
class CommunicatorMock {

public:

  MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::BufGetPageResult>& res, std::string& errMsg));

  MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::SimpleRequestResult>& res, std::string& errMsg));

  MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::BufPinPageResult>& res, std::string& errMsg));

  MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::BufFreezeRequestResult>& res, std::string& errMsg));

};

auto getRandomIndices(int numRequestsPerPage, int numPages) {

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

  return std::move(pageIndices);
}

// this tests getting a page while a return request is being processed.
TEST(BufferManagerFrontendTest, Test6) {

  const UseTemporaryAllocationBlock block(256 * 1024 * 1024);

  const int numRequests = 100;
  const int numPages = 100;
  const int numThreads = 4;

  // the page sizes
  std::vector<size_t> pageSizes = {16, 32, 64};

  // create the frontend
  pdb::PDBBufferManagerFrontEnd frontEnd("tempDSFSD", pageSizes.back(), 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // preallocated the pages
  std::vector<PDBPageHandle> pages;
  pages.reserve(numPages);

  // number of references
  vector<bool> isPageRequested(numPages, false);

  // stuff to sync the threads so they use the frontend in the proper way
  mutex m;
  condition_variable cv;

  auto pageRequestStart = [&](auto &requests, auto pageNumber) {

    // lock the page structure
    unique_lock<mutex> lck(m);

    // wait if there is a request for this page waiting
    cv.wait(lck, [&] { return !isPageRequested[pageNumber] ; });

    // ok we are the only ones making a request
    isPageRequested[pageNumber] = true;
  };

  auto pageRequestEnd = [&] (auto pageNumber) {

    // lock the page structure
    unique_lock<mutex> lck(m);

    // ok we are the only ones making a request
    isPageRequested[pageNumber] = false;

    // send a request to notify that the request is done
    cv.notify_all();
  };

  // init the worker threads of this server
  auto workers = make_shared<PDBWorkerQueue>(make_shared<PDBLogger>("worker.log"), numThreads + 2);

  // create the buzzer
  atomic_int counter;
  counter = 0;
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int& cnt) {
    cnt++;
  });

  // init the frozen vector, this marks which pages are false
  std::vector<atomic_bool> frozen(numPages);
  for_each(frozen.begin(), frozen.end(), [](auto & f) { f = false; });

  // start the threads
  for(int t = 0; t < numThreads; ++t) {

    // grab a worker
    PDBWorkerPtr myWorker = workers->getWorker();

    // the thread
    int thread = t;

    // start the thread
    PDBWorkPtr myWork = make_shared<pdb::GenericWork>([&, thread](PDBBuzzerPtr callerBuzzer) {

      // get the requests
      auto requests = getRandomIndices(numRequests, numPages);

      // make the mock communicator
      auto comm = std::make_shared<CommunicatorMock>();

      // process all requests
      for (int i = 0; i < numRequests * numPages; ++i) {

        /// 1. Request a page

        pageRequestStart(requests, requests[i]);

        // create a get page request
        pdb::Handle<pdb::BufGetPageRequest> pageRequest = pdb::makeObject<pdb::BufGetPageRequest>(std::make_shared<pdb::PDBSet>("db1", "set1"), requests[i]);

        // make sure the mock function returns true
        ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult> &>(), testing::An<std::string &>())).WillByDefault(testing::Invoke(
            [&](pdb::Handle<pdb::BufGetPageResult> &res, std::string &errMsg) {

              // figure out the bytes offset
              auto bytes = (char *) frontEnd.sharedMemory.memory + res->offset;

              // increment the bytes
              for(int j = 0; j < pageSizes[requests[i] % 3]; j += sizeof(int)) {

                // do value stuff
                if(!frozen[requests[i]]) {
                  ((int*) bytes)[j / sizeof(int)] = 0;
                }
                else {
                  ((int*) bytes)[j / sizeof(int)] += j;
                }
              }

              // return true since we assume this succeeded
              return true;
            }
        ));

        // it should call send object exactly once
        EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult> &>(), testing::An<std::string &>())).Times(1);

        // invoke the get page handler
        frontEnd.handleGetPageRequest(pageRequest, comm);

        /// 2. Freeze page

        if(!frozen[requests[i] ]) {

          // make sure the mock function returns true
          ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufFreezeRequestResult> &>(), testing::An<std::string &>())).WillByDefault(testing::Invoke(
              [&](pdb::Handle<pdb::BufFreezeRequestResult> &res, std::string &errMsg) {

                // must be true!
                EXPECT_EQ(res->res, true);

                // return true since we assume this succeeded
                return true;
              }
          ));

          // it should call send object exactly once
          EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufFreezeRequestResult> &>(), testing::An<std::string &>())).Times(1);

          // return page request
          pdb::Handle<pdb::BufFreezeSizeRequest> returnPageRequest = pdb::makeObject<pdb::BufFreezeSizeRequest>("set1", "db1", requests[i], pageSizes[requests[i] % 3]);

          // invoke the return page handler
          frontEnd.handleFreezeSizeRequest(returnPageRequest, comm);

          // freeze it
          frozen[requests[i]] = true;
        }

        /// 3. Return a page

        // make sure the mock function returns true
        ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::SimpleRequestResult> &>(), testing::An<std::string &>())).WillByDefault(testing::Invoke(
            [&](pdb::Handle<pdb::SimpleRequestResult> &res, std::string &errMsg) {

              // must be true!
              EXPECT_EQ(res->getRes().first, true);

              // return true since we assume this succeeded
              return true;
            }
        ));

        // it should call send object exactly once
        EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::SimpleRequestResult> &>(), testing::An<std::string &>())).Times(1);

        // return page request
        pdb::Handle<pdb::BufReturnPageRequest> returnPageRequest = pdb::makeObject<pdb::BufReturnPageRequest>("set1", "db1", requests[i], true);

        // invoke the return page handler
        frontEnd.handleReturnPageRequest(returnPageRequest, comm);

        pageRequestEnd(requests[i]);
      }

      // excellent everything worked just as expected
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    myWorker->execute(myWork, tempBuzzer);
  }

  // wait until all the nodes are finished
  while (counter < numThreads) {
    tempBuzzer->wait();
  }

  for(uint64_t i = 0; i < numPages; ++i) {

    // get the page
    auto page = frontEnd.getPage(make_shared<PDBSet>("db1", "set1"), i);

    for(int j = 0; j < pageSizes[i % 3]; j += sizeof(int)) {

      // init
      ((int*) page->getBytes())[j / sizeof(int)] += j;
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}