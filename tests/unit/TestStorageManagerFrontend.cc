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

// this tests just regular pages
TEST(StorageManagerFrontendTest, Test1) {

  const int numRequests = 1000;
  const int numPages = 100;
  const int pageSize = 64;

  // create the frontend
  pdb::PDBStorageManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // get random seed
  auto seed = (int) std::chrono::system_clock::now().time_since_epoch().count();

  for(uint64_t i = 0; i < numPages; ++i) {

    // get the page
    auto page = frontEnd.getPage(make_shared<PDBSet>("set1", "db1"), i);

    // init the first four bytes of the page to 1;
    for(int j = 0; j < pageSize; j += sizeof(int)) {

      // init
      ((int*) page->getBytes())[j / sizeof(int)] = seed;
    }
  }

  // get the requests
  auto requests = getRandomIndices(numRequests, numPages);

  // make the mock communicator
  auto comm = std::make_shared<CommunicatorMock>();

  for(int i = 0; i < numRequests * numPages; ++i) {

    // create a get page request
    pdb::Handle<pdb::StoGetPageRequest> pageRequest = pdb::makeObject<pdb::StoGetPageRequest>("set1", "db1", requests[i]);

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::StoGetPageResult>& res, std::string& errMsg) {

          // figure out the bytes offset
          auto bytes = (char*) frontEnd.sharedMemory.memory + res->offset;

          for(int j = 0; j < pageSize; j += sizeof(int)) {

            // increment
            ((int*) bytes)[j / sizeof(int)] += j / sizeof(int);
          }

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

    for(int j = 0; j < pageSize; j += sizeof(int)) {

      // check
      EXPECT_EQ(((int*) page->getBytes())[j / sizeof(int)], numRequests * (j / sizeof(int)) + seed);
    }
  }
}

// this tests just regular pages with different sets
TEST(StorageManagerFrontendTest, Test2) {

  const int numRequests = 100;
  const int numPages = 100;
  const int pageSize = 64;
  const int numSets = 2;

  // create the frontend
  pdb::PDBStorageManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // get random seed
  auto seed = (int) std::chrono::system_clock::now().time_since_epoch().count();

  for(uint64_t i = 0; i < numPages; ++i) {

    for(int j = 0; j < numSets; ++j) {

      // figure out the set name
      std::string setName = "set" + std::to_string(j);

      // get the page
      auto page = frontEnd.getPage(make_shared<PDBSet>(setName, "db1"), i);

      // init the first four bytes of the page to 1;
      for(int k = 0; k < pageSize; k += sizeof(int)) {

        // init
        ((int*) page->getBytes())[k / sizeof(int)] = seed;
      }
    }
  }

  // get the requests
  std::vector<std::vector<uint64_t>> setIndices;
  setIndices.reserve(numSets);
  for(int j = 0; j < numSets; ++j) {
    setIndices.emplace_back(getRandomIndices(numRequests, numPages));
  }

  // make the mock communicator
  auto comm = std::make_shared<CommunicatorMock>();

  for(int i = 0; i < numRequests * numPages * numSets; ++i) {

    // create a get page request
    std::string setName = "set" + std::to_string(i % numSets);
    pdb::Handle<pdb::StoGetPageRequest> pageRequest = pdb::makeObject<pdb::StoGetPageRequest>(setName, "db1", setIndices[i % numSets][i / numSets]);

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::StoGetPageResult>& res, std::string& errMsg) {

          // figure out the bytes offset
          auto bytes = (char*) frontEnd.sharedMemory.memory + res->offset;

          for(int j = 0; j < pageSize; j += sizeof(int)) {

            // increment
            ((int*) bytes)[j / sizeof(int)] += j / sizeof(int);
          }

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
    pdb::Handle<pdb::StoReturnPageRequest> returnPageRequest = pdb::makeObject<pdb::StoReturnPageRequest>(setName, "db1", setIndices[i % numSets][i / numSets], true);

    // invoke the return page handler
    frontEnd.handleReturnPageRequest(returnPageRequest, comm);
  }

  for(uint64_t i = 0; i < numPages; ++i) {

    // get the page
    auto page = frontEnd.getPage(make_shared<PDBSet>("set1", "db1"), i);

    for(int j = 0; j < numSets; ++j) {

      for (int k = 0; k < pageSize; k += sizeof(int)) {

        // check
        EXPECT_EQ(((int *) page->getBytes())[k / sizeof(int)], numRequests * (k / sizeof(int)) + seed);
      }
    }
  }
}

// this tests just regular pages
TEST(StorageManagerFrontendTest, Test3) {

  const int numRequests = 1000;
  const int numPages = 100;
  const int pageSize = 64;
  std::vector<int> pageSizes = {16, 32, 64};

  // create the frontend
  pdb::PDBStorageManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // get random seed
  auto seed = (int) std::chrono::system_clock::now().time_since_epoch().count();

  // make the mock communicator
  auto comm = std::make_shared<CommunicatorMock>();

  for(uint64_t i = 0; i < numPages; ++i) {

    /// 1. Get the page init to seed

    // create a get page request
    pdb::Handle<pdb::StoGetPageRequest> pageRequest = pdb::makeObject<pdb::StoGetPageRequest>("set1", "db1", i);

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::StoGetPageResult>& res, std::string& errMsg) {

          // figure out the bytes offset
          auto bytes = (char*) frontEnd.sharedMemory.memory + res->offset;

          // init the first four bytes of the page to 1;
          for(int j = 0; j < pageSize; j += sizeof(int)) {

            // init
            ((int*) bytes)[j / sizeof(int)] = seed;
          }

          // return true since we assume this succeeded
          return true;
        }
    ));

    // it should call send object exactly once
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult>&>(), testing::An<std::string&>())).Times(1);

    // invoke the get page handler
    frontEnd.handleGetPageRequest(pageRequest, comm);

    /// 2. Freeze the size

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

    // make a request to freeze
    pdb::Handle<pdb::StoFreezeSizeRequest> freezePageRequest = pdb::makeObject<pdb::StoFreezeSizeRequest>("set1", "db1", i, pageSizes[i % 3]);
    frontEnd.handleFreezeSizeRequest(freezePageRequest, comm);

    /// 3. Return the page

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
    pdb::Handle<pdb::StoReturnPageRequest> returnPageRequest = pdb::makeObject<pdb::StoReturnPageRequest>("set1", "db1", i, true);

    // invoke the return page handler
    frontEnd.handleReturnPageRequest(returnPageRequest, comm);
  }

  // get the requests
  auto requests = getRandomIndices(numRequests, numPages);

  for(int i = 0; i < numRequests * numPages; ++i) {

    // create a get page request
    pdb::Handle<pdb::StoGetPageRequest> pageRequest = pdb::makeObject<pdb::StoGetPageRequest>("set1", "db1", requests[i]);

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::StoGetPageResult>& res, std::string& errMsg) {

          // figure out the bytes offset
          auto bytes = (char*) frontEnd.sharedMemory.memory + res->offset;

          for(int j = 0; j < pageSizes[requests[i] % 3]; j += sizeof(int)) {

            // increment
            ((int*) bytes)[j / sizeof(int)] += j / sizeof(int);
          }

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

    for(int j = 0; j < pageSizes[i % 3]; j += sizeof(int)) {

      // check
      EXPECT_EQ(((int*) page->getBytes())[j / sizeof(int)], numRequests * (j / sizeof(int)) + seed);
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


}