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

#include <PDBStorageManagerFrontEnd.h>

namespace pdb {

/**
 * This is the mock communicator we provide to the request handlers
 */
class CommunicatorMock {

public:

  MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::StoGetPageResult>& res, std::string& errMsg));

  MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::SimpleRequestResult>& res, std::string& errMsg));

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

TEST(StorageManagerFrontendTest, Test1) {

  const int numRequests = 1000;
  const int numPages = 100;

  // create the frontend
  pdb::PDBStorageManagerFrontEnd frontEnd("tempDSFSD", 64, 2, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  for(uint64_t i = 0; i < numPages; ++i) {

    // get the page
    auto page = frontEnd.getPage(make_shared<PDBSet>("set1", "db1"), i);

    // init the first four bytes of the page to 0;
    *((int*) page->getBytes()) = 1;
  }

  // get the requests
  auto requests = getRandomIndices(numRequests, numPages);

  for(int i = 0; i < numRequests * numPages; ++i) {

    // create a get page request
    pdb::Handle<pdb::StoGetPageRequest> pageRequest = pdb::makeObject<pdb::StoGetPageRequest>("set1", "db1", requests[i]);

    // make the mock communicator
    auto comm = std::make_shared<CommunicatorMock>();

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::StoGetPageResult>& res, std::string& errMsg) {

          // figure out the bytes offset
          auto bytes = (char*) frontEnd.sharedMemory.memory + res->offset;

          // increment the count
          ((int*) bytes)[0]++;

          // return true since we assume this succeeded
          return true;
        }
    ));

    // it should call send object exactly once
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult>&>(), testing::An<std::string&>())).Times(1);

    // invoke the get page handler
    frontEnd.handleGetPageRequest(pageRequest, comm);

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::SimpleRequestResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::SimpleRequestResult>& res, std::string& errMsg) {

          // must be true!
          EXPECT_EQ(res->getRes().first, true);

          // return true since we assume this succeeded
          return true;
        }
    ));

    // it should call send object exactly once
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::SimpleRequestResult>&>(), testing::An<std::string&>())).Times(1);

    // return page request
    pdb::Handle<pdb::StoReturnPageRequest> returnPageRequest = pdb::makeObject<pdb::StoReturnPageRequest>("set1", "db1", requests[i], true);

    // invoke the return page handler
    frontEnd.handleReturnPageRequest(returnPageRequest, comm);
  }

  for(uint64_t i = 0; i < numPages; ++i) {

    // get the page
    auto page = frontEnd.getPage(make_shared<PDBSet>("set1", "db1"), i);

    // init the first four bytes of the page to 0;
    EXPECT_EQ(*((int*) page->getBytes()), numRequests + 1);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


}