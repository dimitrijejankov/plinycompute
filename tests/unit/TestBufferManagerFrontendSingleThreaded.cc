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

// this tests just regular pages
TEST(BufferManagerFrontendTest, Test1) {

  const int numRequests = 1000;
  const int numPages = 100;
  const int pageSize = 64;

  // create the frontend
  pdb::PDBBufferManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // get random seed
  auto seed = (int) std::chrono::system_clock::now().time_since_epoch().count();

  for(uint64_t i = 0; i < numPages; ++i) {

    // get the page
    auto page = frontEnd.getPage(make_shared<PDBSet>("db1", "set1"), i);

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
    pdb::Handle<pdb::BufGetPageRequest> pageRequest = pdb::makeObject<pdb::BufGetPageRequest>(std::make_shared<pdb::PDBSet>("db1", "set1"), requests[i]);

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::BufGetPageResult>& res, std::string& errMsg) {

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
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult>&>(), testing::An<std::string&>())).Times(1);

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
    pdb::Handle<pdb::BufReturnPageRequest> returnPageRequest = pdb::makeObject<pdb::BufReturnPageRequest>("set1", "db1", requests[i], true);

    // invoke the return page handler
    frontEnd.handleReturnPageRequest(returnPageRequest, comm);
  }

  for(uint64_t i = 0; i < numPages; ++i) {

    // get the page
    auto page = frontEnd.getPage(make_shared<PDBSet>("db1", "set1"), i);

    for(int j = 0; j < pageSize; j += sizeof(int)) {

      // check
      EXPECT_EQ(((int*) page->getBytes())[j / sizeof(int)], numRequests * (j / sizeof(int)) + seed);
    }
  }
}

// this tests just regular pages with different sets
TEST(BufferManagerFrontendTest, Test2) {

  const int numRequests = 100;
  const int numPages = 100;
  const int pageSize = 64;
  const int numSets = 2;

  // create the frontend
  pdb::PDBBufferManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // get random seed
  auto seed = (int) std::chrono::system_clock::now().time_since_epoch().count();

  for(uint64_t i = 0; i < numPages; ++i) {

    for(int j = 0; j < numSets; ++j) {

      // figure out the set name
      std::string setName = "set" + std::to_string(j);

      // get the page
      auto page = frontEnd.getPage(make_shared<PDBSet>("db1", setName), i);

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
    pdb::Handle<pdb::BufGetPageRequest> pageRequest = pdb::makeObject<pdb::BufGetPageRequest>(std::make_shared<PDBSet>("db1", setName), setIndices[i % numSets][i / numSets]);

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::BufGetPageResult>& res, std::string& errMsg) {

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
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult>&>(), testing::An<std::string&>())).Times(1);

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
    pdb::Handle<pdb::BufReturnPageRequest> returnPageRequest = pdb::makeObject<pdb::BufReturnPageRequest>(setName, "db1", setIndices[i % numSets][i / numSets], true);

    // invoke the return page handler
    frontEnd.handleReturnPageRequest(returnPageRequest, comm);
  }

  for(uint64_t i = 0; i < numPages; ++i) {

    // get the page
    auto page = frontEnd.getPage(make_shared<PDBSet>("db1", "set1"), i);

    for(int j = 0; j < numSets; ++j) {

      for (int k = 0; k < pageSize; k += sizeof(int)) {

        // check
        EXPECT_EQ(((int *) page->getBytes())[k / sizeof(int)], numRequests * (k / sizeof(int)) + seed);
      }
    }
  }
}

// this tests just regular pages
TEST(BufferManagerFrontendTest, Test3) {

  const int numRequests = 1000;
  const int numPages = 100;
  const int pageSize = 64;
  std::vector<int> pageSizes = {16, 32, 64};

  // create the frontend
  pdb::PDBBufferManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // get random seed
  auto seed = (int) std::chrono::system_clock::now().time_since_epoch().count();

  // make the mock communicator
  auto comm = std::make_shared<CommunicatorMock>();

  for(uint64_t i = 0; i < numPages; ++i) {

    /// 1. Get the page init to seed

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::BufGetPageResult>& res, std::string& errMsg) {

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
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult>&>(), testing::An<std::string&>())).Times(1);

    // create a get page request
    pdb::Handle<pdb::BufGetPageRequest> pageRequest = pdb::makeObject<pdb::BufGetPageRequest>(make_shared<PDBSet>("db1", "set1"), i);

    // invoke the get page handler
    frontEnd.handleGetPageRequest(pageRequest, comm);

    /// 2. Freeze the size

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufFreezeRequestResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::BufFreezeRequestResult>& res, std::string& errMsg) {

          // must be true!
          EXPECT_EQ(res->res, true);

          // return true since we assume this succeeded
          return true;
        }
    ));

    // it should call send object exactly once
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufFreezeRequestResult>&>(), testing::An<std::string&>())).Times(1);

    // make a request to freeze
    pdb::Handle<pdb::BufFreezeSizeRequest> freezePageRequest = pdb::makeObject<pdb::BufFreezeSizeRequest>("set1", "db1", i, pageSizes[i % 3]);
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
    pdb::Handle<pdb::BufReturnPageRequest> returnPageRequest = pdb::makeObject<pdb::BufReturnPageRequest>("set1", "db1", i, true);

    // invoke the return page handler
    frontEnd.handleReturnPageRequest(returnPageRequest, comm);
  }

  // get the requests
  auto requests = getRandomIndices(numRequests, numPages);

  for(int i = 0; i < numRequests * numPages; ++i) {

    // create a get page request
    pdb::Handle<pdb::BufGetPageRequest> pageRequest = pdb::makeObject<pdb::BufGetPageRequest>(make_shared<PDBSet>("db1", "set1"), requests[i]);

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::BufGetPageResult>& res, std::string& errMsg) {

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
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult>&>(), testing::An<std::string&>())).Times(1);

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
    pdb::Handle<pdb::BufReturnPageRequest> returnPageRequest = pdb::makeObject<pdb::BufReturnPageRequest>("set1", "db1", requests[i], true);

    // invoke the return page handler
    frontEnd.handleReturnPageRequest(returnPageRequest, comm);
  }

  for(uint64_t i = 0; i < numPages; ++i) {

    // get the page
    auto page = frontEnd.getPage(make_shared<PDBSet>("db1", "set1"), i);

    for(int j = 0; j < pageSizes[i % 3]; j += sizeof(int)) {

      // check
      EXPECT_EQ(((int*) page->getBytes())[j / sizeof(int)], numRequests * (j / sizeof(int)) + seed);
    }
  }
}

// this test tests anonymous pages
TEST(BufferManagerFrontendTest, Test4) {

  const int numPages = 1000;
  const int pageSize = 64;

  // create the frontend
  pdb::PDBBufferManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // make the mock communicator
  auto comm = std::make_shared<CommunicatorMock>();

  std::vector<uint64_t> pageNumbers;
  pageNumbers.reserve(numPages);

  for(int i = 0; i < numPages; ++i) {

    /// 1. Grab an anonymous page

    // make sure the mock function returns true, write something to the page and
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::BufGetPageResult>& res, std::string& errMsg) {

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
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult>&>(), testing::An<std::string&>())).Times(1);

    // create a get page request
    pdb::Handle<pdb::BufGetAnonymousPageRequest> pageRequest = pdb::makeObject<pdb::BufGetAnonymousPageRequest>(pageSize);

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
    pdb::Handle<pdb::BufUnpinPageRequest> unpinRequest = pdb::makeObject<pdb::BufUnpinPageRequest>(nullptr, pageNumbers.back(), true);

    // invoke the get page handler
    frontEnd.handleUnpinPageRequest(unpinRequest, comm);
  }

  // go through each page
  int counter = 0;
  for(auto page : pageNumbers) {

    /// 1. Pin the page

    // make sure the mock function returns true
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufPinPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::BufPinPageResult>& res, std::string& errMsg) {

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
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufPinPageResult>&>(), testing::An<std::string&>())).Times(1);

    // create a get page request
    pdb::Handle<pdb::BufPinPageRequest> pinRequest = pdb::makeObject<pdb::BufPinPageRequest>(nullptr, page);

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
    pdb::Handle<pdb::BufUnpinPageRequest> unpinRequest = pdb::makeObject<pdb::BufUnpinPageRequest>(nullptr, page, false);

    // invoke the get page handler
    frontEnd.handleUnpinPageRequest(unpinRequest, comm);

    // increment it
    counter++;
  }
}

// this test tests anonymous pages with different page sizes
TEST(BufferManagerFrontendTest, Test5) {

  const int numPages = 1000;
  const int pageSize = 64;
  std::vector<int> pageSizes = {16, 32, 64};

  // create the frontend
  pdb::PDBBufferManagerFrontEnd frontEnd("tempDSFSD", pageSize, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // make the mock communicator
  auto comm = std::make_shared<CommunicatorMock>();

  std::vector<uint64_t> pageNumbers;
  pageNumbers.reserve(numPages);

  for(int i = 0; i < numPages; ++i) {

    /// 1. Grab an anonymous page

    // make sure the mock function returns true, write something to the page and
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::BufGetPageResult>& res, std::string& errMsg) {

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
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufGetPageResult>&>(), testing::An<std::string&>())).Times(1);

    // create a get page request
    pdb::Handle<pdb::BufGetAnonymousPageRequest> pageRequest = pdb::makeObject<pdb::BufGetAnonymousPageRequest>(pageSizes[i % 3]);

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
      pdb::Handle<pdb::BufUnpinPageRequest> unpinRequest = pdb::makeObject<pdb::BufUnpinPageRequest>(nullptr, pageNumbers.back(), true);

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
      pdb::Handle<pdb::BufReturnAnonPageRequest> returnPageRequest = pdb::makeObject<pdb::BufReturnAnonPageRequest>(pageNumbers.back(), true);

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
    ON_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufPinPageResult>&>(), testing::An<std::string&>())).WillByDefault(testing::Invoke(
        [&] (pdb::Handle<pdb::BufPinPageResult>& res, std::string& errMsg) {

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
    EXPECT_CALL(*comm, sendObject(testing::An<pdb::Handle<pdb::BufPinPageResult>&>(), testing::An<std::string&>())).Times(1);

    // create a get page request
    pdb::Handle<pdb::BufPinPageRequest> pinRequest = pdb::makeObject<pdb::BufPinPageRequest>(nullptr, page);

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
    pdb::Handle<pdb::BufUnpinPageRequest> unpinRequest = pdb::makeObject<pdb::BufUnpinPageRequest>(nullptr, page, false);

    // invoke the get page handler
    frontEnd.handleUnpinPageRequest(unpinRequest, comm);

    // increment it
    counter++;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}