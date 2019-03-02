//
// Created by dimitrije on 2/1/19.
//


#include <PDBBufferManagerBackEnd.h>
#include "TestBufferManagerBackend.h"

namespace pdb {

// this test checks whether anonymous pages work on the backend
TEST(BufferManagerBackendTest, Test1) {

  const size_t numPages = 100;
  const size_t pageSize = 64;

  int curPage = 0;
  vector<bool> pinned(numPages, false);
  vector<bool> frozen(numPages, false);
  std::unordered_map<int64_t, int64_t> pages;

  // allocate memory
  std::unique_ptr<char[]> memory(new char[numPages * pageSize]);

  // make the shared memory object
  PDBSharedMemory sharedMemory{};
  sharedMemory.pageSize = pageSize;
  sharedMemory.numPages = numPages;
  sharedMemory.memory = memory.get();

  pdb::PDBBufferManagerBackEnd<MockRequestFactory> bufferManager(sharedMemory);

  MockRequestFactory::_requestFactory = std::make_shared<MockRequestFactoryImpl>();

  MockServer server;
  ON_CALL(server, getConfiguration).WillByDefault(testing::Invoke(
      [&]() {
        return std::make_shared<pdb::NodeConfig>();
      }));

  EXPECT_CALL(server, getConfiguration).Times(testing::AtLeast(1));

  bufferManager.recordServer(server);

  /// 1. Mock the anonymous pages request

  ON_CALL(*MockRequestFactory::_requestFactory, getAnonPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger,
          int port,
          const std::string &address,
          pdb::PDBPageHandle onErr,
          size_t bytesForRequest,
          const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::BufGetPageResult>)> &processResponse,
          size_t minSize) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        int64_t myPage = curPage++;

        // make the page
        pdb::Handle<pdb::BufGetPageResult> returnPageRequest =
            pdb::makeObject<pdb::BufGetPageResult>(myPage * pageSize, myPage, true, false, -1, pageSize, "", "");

        // mark it as pinned
        pinned[myPage] = true;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly 98 times
  EXPECT_CALL(*MockRequestFactory::_requestFactory, getAnonPage).Times(98);

  /// 2. Mock the unpin page

  ON_CALL(*MockRequestFactory::_requestFactory, unpinPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, bool onErr, size_t bytesForRequest,
          const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
          PDBSetPtr &set, size_t pageNum, bool isDirty) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);

        // make the page
        pdb::Handle<pdb::SimpleRequestResult> returnPageRequest = pdb::makeObject<pdb::SimpleRequestResult>(true, "");

        // expect it to be pinned when you return it
        EXPECT_TRUE(pinned[pageNum]);

        // mark it as unpinned
        pinned[pageNum] = false;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly twice
  EXPECT_CALL(*MockRequestFactory::_requestFactory, unpinPage).Times(2);

  /// 3. Mock the freeze size

  ON_CALL(*MockRequestFactory::_requestFactory, freezeSize).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string address, bool onErr,
          size_t bytesForRequest, const std::function<bool(pdb::Handle<pdb::BufFreezeRequestResult>)> &processResponse,
          pdb::PDBSetPtr setPtr, size_t pageNum, size_t numBytes) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);

        // make the page
        pdb::Handle<pdb::BufFreezeRequestResult> returnPageRequest = pdb::makeObject<pdb::BufFreezeRequestResult>(true);

        // expect not to be frozen
        EXPECT_FALSE(frozen[pageNum]);

        // mark it as frozen
        frozen[pageNum] = true;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly once
  EXPECT_CALL(*MockRequestFactory::_requestFactory, freezeSize).Times(1);

  /// 4. Mock the pin page

  ON_CALL(*MockRequestFactory::_requestFactory, pinPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, bool onErr,
          size_t bytesForRequest, const std::function<bool(pdb::Handle<pdb::BufPinPageResult>)> &processResponse,
          const pdb::PDBSetPtr &setPtr, size_t pageNum) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);

        // make the page
        pdb::Handle<pdb::BufPinPageResult>
            returnPageRequest = pdb::makeObject<pdb::BufPinPageResult>(pageNum * pageSize, true);

        // mark it as unpinned
        pinned[pageNum] = true;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly twice
  EXPECT_CALL(*MockRequestFactory::_requestFactory, pinPage).Times(2);

  /// 5. Mock return anon page

  ON_CALL(*MockRequestFactory::_requestFactory, returnAnonPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, bool onErr, size_t bytesForRequest,
          const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
          size_t pageNum, bool isDirty) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);

        // make the page
        pdb::Handle<pdb::SimpleRequestResult> returnPageRequest = pdb::makeObject<pdb::SimpleRequestResult>(true, "");

        // expect it to be pinned when you return it
        EXPECT_TRUE(pinned[pageNum]);

        // mark it as unpinned
        pinned[pageNum] = false;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly 98 times
  EXPECT_CALL(*MockRequestFactory::_requestFactory, returnAnonPage).Times(98);

  {

    // grab two pages
    pdb::PDBPageHandle page1 = bufferManager.getPage();
    pdb::PDBPageHandle page2 = bufferManager.getPage();

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

    // check whether we are null
    EXPECT_EQ(page1->getBytes(), nullptr);
    EXPECT_FALSE(pinned[page1->whichPage()]);

    // freeze the size to 32 and unpin it
    page2->freezeSize(32);
    page2->unpin();

    // check whether we are null
    EXPECT_EQ(page2->getBytes(), nullptr);
    EXPECT_FALSE(pinned[page2->whichPage()]);

    // just grab some random pages
    for (int i = 0; i < 32; i++) {
      pdb::PDBPageHandle page3 = bufferManager.getPage();
      pdb::PDBPageHandle page4 = bufferManager.getPage();
      pdb::PDBPageHandle page5 = bufferManager.getPage();
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

  // just to remove the mock object
  MockRequestFactory::_requestFactory = nullptr;
  bufferManager.parent = nullptr;
}

// tests set pages
TEST(BufferManagerBackendTest, Test2) {

  const size_t numPages = 1000;
  const size_t numSets = 2;
  const size_t pageSize = 64;

  vector<bool> pinned(numPages * numSets, false);
  vector<bool> frozen(numPages * numSets, false);
  std::unordered_map<int64_t, int64_t> pages;

  // allocate memory
  std::unique_ptr<char[]> memory(new char[numPages * pageSize * numSets]);

  // make the shared memory object
  PDBSharedMemory sharedMemory{};
  sharedMemory.pageSize = pageSize;
  sharedMemory.numPages = numPages * numSets;
  sharedMemory.memory = memory.get();

  pdb::PDBBufferManagerBackEnd<MockRequestFactory> myMgr(sharedMemory);

  MockRequestFactory::_requestFactory = std::make_shared<MockRequestFactoryImpl>();

  MockServer server;
  ON_CALL(server, getConfiguration).WillByDefault(testing::Invoke(
      [&]() {
        return std::make_shared<pdb::NodeConfig>();
      }));

  EXPECT_CALL(server, getConfiguration).Times(testing::AtLeast(1));

  myMgr.recordServer(server);

  /// 1. Mock the get page for the set

  ON_CALL(*MockRequestFactory::_requestFactory, getPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger,
          int port,
          const std::string &address,
          pdb::PDBPageHandle onErr,
          size_t bytesForRequest,
          const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::BufGetPageResult>)> &processResponse,
          pdb::PDBSetPtr set,
          uint64_t pageNum) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);
        EXPECT_TRUE(set->getSetName() == "set1" || set->getSetName() == "set2");
        EXPECT_TRUE(set->getDBName() == "DB");

        // figure out which set
        int whichSet = set->getSetName() == "set1" ? 0 : 1;

        // make the page
        pdb::Handle<pdb::BufGetPageResult> returnPageRequest = pdb::makeObject<pdb::BufGetPageResult>((whichSet * numPages + pageNum) * pageSize, pageNum, false, false, -1, pageSize, set->getSetName(), set->getDBName());

        // mark it as pinned
        pinned[whichSet * numPages + pageNum] = true;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly 98 times
  EXPECT_CALL(*MockRequestFactory::_requestFactory, getPage).Times(98);


  /// 2. Mock the unpin page

  ON_CALL(*MockRequestFactory::_requestFactory, unpinPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, bool onErr, size_t bytesForRequest,
          const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
          PDBSetPtr &set, size_t pageNum, bool isDirty) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);

        EXPECT_TRUE(set->getSetName() == "set1" || set->getSetName() == "set2");
        EXPECT_TRUE(set->getDBName() == "DB");

        // figure out which set
        int whichSet = set->getSetName() == "set1" ? 0 : 1;

        // make the page
        pdb::Handle<pdb::SimpleRequestResult> returnPageRequest = pdb::makeObject<pdb::SimpleRequestResult>(true, "");

        // expect it to be pinned when you return it
        EXPECT_TRUE(pinned[whichSet * numPages + pageNum]);

        // mark it as unpinned
        pinned[whichSet * numPages + pageNum] = false;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly twice
  EXPECT_CALL(*MockRequestFactory::_requestFactory, unpinPage).Times(2);

  /// 3. Mock the freeze size

  ON_CALL(*MockRequestFactory::_requestFactory, freezeSize).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string address, bool onErr,
          size_t bytesForRequest, const std::function<bool(pdb::Handle<pdb::BufFreezeRequestResult>)> &processResponse,
          pdb::PDBSetPtr setPtr, size_t pageNum, size_t numBytes) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);
        EXPECT_TRUE(setPtr->getSetName() == "set1" || setPtr->getSetName() == "set2");
        EXPECT_TRUE(setPtr->getDBName() == "DB");

        // figure out which set
        int whichSet = setPtr->getSetName() == "set1" ? 0 : 1;

        // make the page
        pdb::Handle<pdb::BufFreezeRequestResult> returnPageRequest = pdb::makeObject<pdb::BufFreezeRequestResult>(true);

        // expect not to be frozen
        EXPECT_FALSE(frozen[whichSet * numPages + pageNum]);

        // mark it as frozen
        frozen[whichSet * numPages + pageNum] = true;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // expected to be called 33 times
  EXPECT_CALL(*MockRequestFactory::_requestFactory, freezeSize).Times(33);

  /// 4. Mock the pin page

  ON_CALL(*MockRequestFactory::_requestFactory, pinPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, bool onErr,
          size_t bytesForRequest, const std::function<bool(pdb::Handle<pdb::BufPinPageResult>)> &processResponse,
          const pdb::PDBSetPtr &setPtr, size_t pageNum) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);
        EXPECT_TRUE(setPtr->getSetName() == "set1" || setPtr->getSetName() == "set2");
        EXPECT_TRUE(setPtr->getDBName() == "DB");

        // figure out which set
        int whichSet = setPtr->getSetName() == "set1" ? 0 : 1;

        // make the page
        pdb::Handle<pdb::BufPinPageResult> returnPageRequest = pdb::makeObject<pdb::BufPinPageResult>((whichSet * numPages + pageNum) * pageSize, true);

        // mark it as unpinned
        pinned[whichSet * numPages + pageNum] = true;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // expected to be called twice
  EXPECT_CALL(*MockRequestFactory::_requestFactory, pinPage).Times(2);

  /// 5. Mock return page

  ON_CALL(*MockRequestFactory::_requestFactory, returnPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, bool onErr,
          size_t bytesForRequest, const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
          std::string setName, std::string dbName, size_t pageNum, bool isDirty) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);
        EXPECT_TRUE(setName == "set1" || setName == "set2");
        EXPECT_TRUE(dbName == "DB");

        // make the page
        pdb::Handle<pdb::SimpleRequestResult> returnPageRequest = pdb::makeObject<pdb::SimpleRequestResult>(true, "");

        // figure out which set
        int whichSet = setName == "set1" ? 0 : 1;

        // expect it to be pinned when you return it
        EXPECT_TRUE(pinned[whichSet * numPages + pageNum]);

        // mark it as unpinned
        pinned[whichSet * numPages + pageNum] = false;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly 98 times
  EXPECT_CALL(*MockRequestFactory::_requestFactory, returnPage).Times(98);

  {
    PDBSetPtr set1 = make_shared<PDBSet>("DB", "set1");
    PDBSetPtr set2 = make_shared<PDBSet>("DB", "set2");
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
    for (uint64_t i = 0; i < 32; i++) {
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

  // just to remove the mock object
  MockRequestFactory::_requestFactory = nullptr;
  myMgr.parent = nullptr;
}

}