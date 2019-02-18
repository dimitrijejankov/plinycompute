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

};

TEST(BufferManagerFrontendTest, Test7) {

  const UseTemporaryAllocationBlock block(256 * 1024 * 1024);

  const int numPages = 1000;
  const int pageSize = 64;
  const int numThreads = 8;

  // create the frontend
  pdb::PDBBufferManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // stuff to sync the threads
  mutex m;

  // init the worker threads of this server
  auto workers = make_shared<PDBWorkerQueue>(make_shared<PDBLogger>("worker.log"), numThreads + 2);

  // create the buzzer
  atomic_int counter;
  counter = 0;
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int& cnt) {
    cnt++;
  });

  // start the threads
  for(int t = 0; t < numThreads; ++t) {

    // grab a worker
    PDBWorkerPtr myWorker = workers->getWorker();

    // the thread
    int thread = t;

    // start the thread
    PDBWorkPtr myWork = make_shared<pdb::GenericWork>([&, thread](PDBBuzzerPtr callerBuzzer) {

      // make the mock communicator
      auto comm = std::make_shared<CommunicatorMock>();

      // where we put the pages
      std::vector<uint64_t> pages;
      pages.reserve(numPages);

      // process all requests
      for (int i = 0; i < numPages; ++i) {

        /// 1. Request an anonymous page

        // create a get page request
        pdb::Handle<pdb::BufGetAnonymousPageRequest> pageRequest = pdb::makeObject<pdb::BufGetAnonymousPageRequest>(pageSize);

        // make sure the mock function returns true
        ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult> &>(), testing::An<std::string &>())).WillByDefault(testing::Invoke(
            [&](pdb::Handle<pdb::BufGetPageResult> &res, std::string &errMsg) {

              // figure out the bytes offset
              auto bytes = (char *) frontEnd.sharedMemory.memory + res->offset;

              // fill up the age
              ((int*) bytes)[0] = i + 1;
              ((int*) bytes)[1] = i + 2;
              ((int*) bytes)[2] = i + 3;
              ((int*) bytes)[3] = i + 4;

              // store the page number
              pages.emplace_back(res->pageNum);

              // return true since we assume this succeeded
              return true;
            }
        ));

        // it should call send object exactly once
        EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult> &>(), testing::An<std::string &>())).Times(1);

        // invoke the get page handler
        frontEnd.handleGetAnonymousPageRequest(pageRequest, comm);

        /// 2. Unpin the page

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
        pdb::Handle<pdb::BufUnpinPageRequest> returnPageRequest = pdb::makeObject<pdb::BufUnpinPageRequest>(nullptr, pages.back(), true);

        // invoke the return page handler
        frontEnd.handleUnpinPageRequest(returnPageRequest, comm);
      }

      // check if everything is fine
      int cnt = 0;
      for (auto page : pages) {

        /// 3. Repin the page

        // create a repin request
        pdb::Handle<pdb::BufPinPageRequest> pageRequest = pdb::makeObject<pdb::BufPinPageRequest>(nullptr, page);

        // make sure the mock function returns true
        ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufPinPageResult> &>(), testing::An<std::string &>())).WillByDefault(testing::Invoke(
            [&](pdb::Handle<pdb::BufPinPageResult> &res, std::string &errMsg) {

              // figure out the bytes offset
              auto bytes = (char *) frontEnd.sharedMemory.memory + res->offset;

              // fill up the age
              ((int*) bytes)[0] = cnt + 1;
              ((int*) bytes)[1] = cnt + 2;
              ((int*) bytes)[2] = cnt + 3;
              ((int*) bytes)[3] = cnt + 4;

              // return true since we assume this succeeded
              return true;
            }
        ));

        // it should call send object exactly once
        EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufPinPageResult> &>(), testing::An<std::string &>())).Times(1);

        // handle the pin page request
        frontEnd.handlePinPageRequest(pageRequest, comm);

        /// 4. Return the page

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
        pdb::Handle<pdb::BufReturnAnonPageRequest> returnPageRequest = pdb::makeObject<pdb::BufReturnAnonPageRequest>(page, false);

        // invoke the return page handler
        frontEnd.handleReturnAnonPageRequest(returnPageRequest, comm);

        // increment the count
        cnt++;
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
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}