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
#include <GenericWork.h>

namespace pdb {

/**
 * This is the mock communicator we provide to the request handlers
 */
class CommunicatorMock {

public:

  MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::StoGetPageResult>& res, std::string& errMsg));

  MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::SimpleRequestResult>& res, std::string& errMsg));

  MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::StoPinPageResult>& res, std::string& errMsg));

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

    // create a get page request
    pdb::Handle<pdb::StoGetPageRequest> pageRequest = pdb::makeObject<pdb::StoGetPageRequest>("set1", "db1", i);

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

// this test tests anonymous pages
TEST(StorageManagerFrontendTest, Test4) {

  const int numPages = 1000;
  const int pageSize = 64;

  // create the frontend
  pdb::PDBStorageManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // make the mock communicator
  auto comm = std::make_shared<CommunicatorMock>();

  std::vector<uint64_t> pageNumbers;
  pageNumbers.reserve(numPages);

  for(int i = 0; i < numPages; ++i) {

    /// 1. Grab an anonymous page

    // make sure the mock function returns true, write something to the page and
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::StoGetPageResult>& res, std::string& errMsg) {

          // check if it actually is an anonymous page
          EXPECT_TRUE(res->isAnonymous);
          EXPECT_EQ(MIN_PAGE_SIZE << res->numBytes, pageSize);
          EXPECT_TRUE(res->setName.operator==(""));
          EXPECT_TRUE(res->dbName.operator==(""));

          // figure out the bytes offset
          auto bytes = (char*) frontEnd.sharedMemory.memory + res->offset;

          // fill up the age
          for(int j = 0; j < pageSize; j += sizeof(int)) {

            // increment
            ((int*) bytes)[j / sizeof(int)] = (j + i) / sizeof(int);
          }

          // store the page number
          pageNumbers.emplace_back(res->pageNum);

          // return true since we assume this succeeded
          return true;
        }
    ));

    // it should call send object exactly once
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult>&>(), testing::An<std::string&>())).Times(1);

    // create a get page request
    pdb::Handle<pdb::StoGetAnonymousPageRequest> pageRequest = pdb::makeObject<pdb::StoGetAnonymousPageRequest>(pageSize);

    // invoke the get page handler
    frontEnd.handleGetAnonymousPageRequest(pageRequest, comm);


    /// 2. Unpin the anonymous page

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

    // create a unpin request
    pdb::Handle<pdb::StoUnpinPageRequest> unpinRequest = pdb::makeObject<pdb::StoUnpinPageRequest>(nullptr, pageNumbers.back(), true);

    // invoke the get page handler
    frontEnd.handleUnpinPageRequest(unpinRequest, comm);
  }

  // go through each page
  int counter = 0;
  for(auto page : pageNumbers) {

    /// 1. Pin the page

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoPinPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::StoPinPageResult>& res, std::string& errMsg) {

          // make sure the pin succeeded
          EXPECT_TRUE(res->success);

          // figure out the bytes offset
          auto bytes = (char*) frontEnd.sharedMemory.memory + res->offset;

          // fill up the age
          for(int j = 0; j < pageSize; j += sizeof(int)) {

            // check if this is equal
            EXPECT_EQ(((int*) bytes)[j / sizeof(int)], (j + counter) / sizeof(int));
          }

          // return true since we assume this succeeded
          return true;
        }
    ));

    // it should call send object exactly once
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoPinPageResult>&>(), testing::An<std::string&>())).Times(1);

    // create a get page request
    pdb::Handle<pdb::StoPinPageRequest> pinRequest = pdb::makeObject<pdb::StoPinPageRequest>(nullptr, page);

    // invoke the get page handler
    frontEnd.handlePinPageRequest(pinRequest, comm);

    /// 2. Unpin the anonymous page

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

    // create a unpin request
    pdb::Handle<pdb::StoUnpinPageRequest> unpinRequest = pdb::makeObject<pdb::StoUnpinPageRequest>(nullptr, page, false);

    // invoke the get page handler
    frontEnd.handleUnpinPageRequest(unpinRequest, comm);

    // increment it
    counter++;
  }
}

// this test tests anonymous pages with different page sizes
TEST(StorageManagerFrontendTest, Test5) {

  const int numPages = 1000;
  const int pageSize = 64;
  std::vector<int> pageSizes = {16, 32, 64};

  // create the frontend
  pdb::PDBStorageManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // make the mock communicator
  auto comm = std::make_shared<CommunicatorMock>();

  std::vector<uint64_t> pageNumbers;
  pageNumbers.reserve(numPages);

  for(int i = 0; i < numPages; ++i) {

    /// 1. Grab an anonymous page

    // make sure the mock function returns true, write something to the page and
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::StoGetPageResult>& res, std::string& errMsg) {

          // check if it actually is an anonymous page
          EXPECT_TRUE(res->isAnonymous);
          EXPECT_EQ(MIN_PAGE_SIZE << res->numBytes, pageSizes[i % 3]);
          EXPECT_TRUE(res->setName.operator==(""));
          EXPECT_TRUE(res->dbName.operator==(""));

          // figure out the bytes offset
          auto bytes = (char*) frontEnd.sharedMemory.memory + res->offset;

          // fill up the age
          for(int j = 0; j < pageSizes[i % 3]; j += sizeof(int)) {

            // increment
            ((int*) bytes)[j / sizeof(int)] = (j + i) / sizeof(int);
          }

          // store the page number
          pageNumbers.emplace_back(res->pageNum);

          // return true since we assume this succeeded
          return true;
        }
    ));

    // it should call send object exactly once
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult>&>(), testing::An<std::string&>())).Times(1);

    // create a get page request
    pdb::Handle<pdb::StoGetAnonymousPageRequest> pageRequest = pdb::makeObject<pdb::StoGetAnonymousPageRequest>(pageSizes[i % 3]);

    // invoke the get page handler
    frontEnd.handleGetAnonymousPageRequest(pageRequest, comm);


    /// 2. if i is even unpin the anonymous page, if i is odd return the anonymous page

    if(i % 2 == 0) {

      /// 2.1 Unpin

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

      // create a unpin request
      pdb::Handle<pdb::StoUnpinPageRequest> unpinRequest = pdb::makeObject<pdb::StoUnpinPageRequest>(nullptr, pageNumbers.back(), true);

      // invoke the get page handler
      frontEnd.handleUnpinPageRequest(unpinRequest, comm);
    }
    else {

      /// 2.2 Return page

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
      pdb::Handle<pdb::StoReturnAnonPageRequest> returnPageRequest = pdb::makeObject<pdb::StoReturnAnonPageRequest>(pageNumbers.back(), true);

      // invoke the return page handler
      frontEnd.handleReturnAnonPageRequest(returnPageRequest, comm);

      // remove the page
      pageNumbers.pop_back();
    }
  }

  // go through each page
  int counter = 0;
  for(auto page : pageNumbers) {

    /// 1. Pin the page

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoPinPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::StoPinPageResult>& res, std::string& errMsg) {

          // make sure the pin succeeded
          EXPECT_TRUE(res->success);

          // figure out the bytes offset
          auto bytes = (char*) frontEnd.sharedMemory.memory + res->offset;

          // fill up the age
          for(int j = 0; j < pageSizes[(counter * 2) % 3]; j += sizeof(int)) {

            // check if this is equal
            EXPECT_EQ(((int*) bytes)[j / sizeof(int)], (j + (counter * 2)) / sizeof(int));
          }

          // return true since we assume this succeeded
          return true;
        }
    ));

    // it should call send object exactly once
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoPinPageResult>&>(), testing::An<std::string&>())).Times(1);

    // create a get page request
    pdb::Handle<pdb::StoPinPageRequest> pinRequest = pdb::makeObject<pdb::StoPinPageRequest>(nullptr, page);

    // invoke the get page handler
    frontEnd.handlePinPageRequest(pinRequest, comm);

    /// 2. Unpin the anonymous page

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

    // create a unpin request
    pdb::Handle<pdb::StoUnpinPageRequest> unpinRequest = pdb::makeObject<pdb::StoUnpinPageRequest>(nullptr, page, false);

    // invoke the get page handler
    frontEnd.handleUnpinPageRequest(unpinRequest, comm);

    // increment it
    counter++;
  }
}

/*
// this tests getting a page while a return request is being processed.
TEST(StorageManagerFrontendTest, Test6) {

  const UseTemporaryAllocationBlock block(256 * 1024 * 1024);

  const int numRequests = 1000;
  const int numPages = 100;
  const int pageSize = 64;
  const int numThreads = 4;

  // create the frontend
  pdb::PDBStorageManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // get random seed
  //auto seed = (int) std::chrono::system_clock::now().time_since_epoch().count();
  auto seed = 0;

  for(uint64_t i = 0; i < numPages; ++i) {

    // get the page
    auto page = frontEnd.getPage(make_shared<PDBSet>("set1", "db1"), i);

    // init the first four bytes of the page to 1;
    for(int t = 0; t < numThreads; ++t) {

      // init
      ((int*) page->getBytes())[t] = seed;
    }
  }

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
  int counter = 0;
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, int& cnt) {
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

      // get the requests
      auto requests = getRandomIndices(numRequests, numPages);

      // make the mock communicator
      auto comm = std::make_shared<CommunicatorMock>();

      // process all requests
      for (int i = 0; i < numRequests * numPages; ++i) {

        /// 1. Request a page

        pageRequestStart(requests, requests[i]);

        // create a get page request
        pdb::Handle<pdb::StoGetPageRequest> pageRequest = pdb::makeObject<pdb::StoGetPageRequest>("set1", "db1", requests[i]);

        // make sure the mock function returns true
        ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult> &>(), testing::An<std::string &>())).WillByDefault(testing::Invoke(
            [&](pdb::Handle<pdb::StoGetPageResult> &res, std::string &errMsg) {

              // figure out the bytes offset
              auto bytes = (char *) frontEnd.sharedMemory.memory + res->offset;

              // increment the bytes
              ((int *) bytes)[thread] += 1;

              // return true since we assume this succeeded
              return true;
            }
        ));

        // it should call send object exactly once
        EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult> &>(), testing::An<std::string &>())).Times(1);

        // invoke the get page handler
        frontEnd.handleGetPageRequest(pageRequest, comm);

        /// 2. Return a page

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
        pdb::Handle<pdb::StoReturnPageRequest> returnPageRequest = pdb::makeObject<pdb::StoReturnPageRequest>("set1", "db1", requests[i], true);

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
    auto page = frontEnd.getPage(make_shared<PDBSet>("set1", "db1"), i);

    for(int t = 0; t < numThreads; ++t) {

      // check
      EXPECT_EQ(((int*) page->getBytes())[t], numRequests + seed);
    }
  }
}*/

TEST(StorageManagerFrontendTest, Test7) {

  const UseTemporaryAllocationBlock block(256 * 1024 * 1024);

  const int numPages = 1000;
  const int pageSize = 64;
  const int numThreads = 8;

  // create the frontend
  pdb::PDBStorageManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // stuff to sync the threads
  mutex m;

  // init the worker threads of this server
  auto workers = make_shared<PDBWorkerQueue>(make_shared<PDBLogger>("worker.log"), numThreads + 2);

  // create the buzzer
  int counter = 0;
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, int& cnt) {
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
        pdb::Handle<pdb::StoGetAnonymousPageRequest> pageRequest = pdb::makeObject<pdb::StoGetAnonymousPageRequest>(pageSize);

        // make sure the mock function returns true
        ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult> &>(), testing::An<std::string &>())).WillByDefault(testing::Invoke(
            [&](pdb::Handle<pdb::StoGetPageResult> &res, std::string &errMsg) {

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
        EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoGetPageResult> &>(), testing::An<std::string &>())).Times(1);

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
        pdb::Handle<pdb::StoUnpinPageRequest> returnPageRequest = pdb::makeObject<pdb::StoUnpinPageRequest>(nullptr, pages.back(), true);

        // invoke the return page handler
        frontEnd.handleUnpinPageRequest(returnPageRequest, comm);
      }

      // check if everything is fine
      int cnt = 0;
      for (auto page : pages) {

        /// 3. Repin the page

        // create a repin request
        pdb::Handle<pdb::StoPinPageRequest> pageRequest = pdb::makeObject<pdb::StoPinPageRequest>(nullptr, page);

        // make sure the mock function returns true
        ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoPinPageResult> &>(), testing::An<std::string &>())).WillByDefault(testing::Invoke(
            [&](pdb::Handle<pdb::StoPinPageResult> &res, std::string &errMsg) {

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
        EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::StoPinPageResult> &>(), testing::An<std::string &>())).Times(1);

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
        pdb::Handle<pdb::StoReturnAnonPageRequest> returnPageRequest = pdb::makeObject<pdb::StoReturnAnonPageRequest>(page, false);

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