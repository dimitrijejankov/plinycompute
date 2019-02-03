//
// Created by dimitrije on 2/1/19.
//

#include <gtest/gtest.h>
#include <PDBStorageManagerBackEnd.h>
#include <gmock/gmock.h>

namespace pdb {

class MockServer : public pdb::PDBServer {
public:

  MOCK_METHOD0(getConfiguration, pdb::NodeConfigPtr());

  // mark the tests for the backend
  FRIEND_TEST(StorageManagerBackendTest, Test1);
};

class MockRequestFactoryImpl {
public:

MOCK_METHOD9(getPage, pdb::PDBPageHandle(pdb::PDBLoggerPtr &myLogger,
                                         int port,
                                         const std::string address,
                                         pdb::PDBPageHandle onErr,
                                         size_t bytesForRequest,
                                         const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::StoGetPageResult>)> &processResponse,
                                         std::string setName,
                                         std::string dbName,
                                         uint64_t pageNum));

MOCK_METHOD7(getAnonPage, pdb::PDBPageHandle(pdb::PDBLoggerPtr &myLogger,
                                             int port,
                                             const std::string &address,
                                             pdb::PDBPageHandle onErr,
                                             size_t bytesForRequest,
                                             const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::StoGetPageResult>)> &processResponse,
                                             size_t minSize));

MOCK_METHOD9(unpinPage, bool(pdb::PDBLoggerPtr &myLogger,
                               int port,
                               const std::string &address,
                               bool onErr,
                               size_t bytesForRequest,
                               const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
                               PDBSetPtr &set,
                               size_t pageNum,
                               bool isDirty));

MOCK_METHOD10(returnPage, bool(pdb::PDBLoggerPtr &myLogger,
                              int port,
                              const std::string &address,
                              bool onErr,
                              size_t bytesForRequest,
                              const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
                              std::string setName,
                              std::string dbName,
                              size_t pageNum,
                              bool isDirty));

MOCK_METHOD8(returnAnonPage, bool(pdb::PDBLoggerPtr &myLogger,
                                  int port,
                                  const std::string &address,
                                  bool onErr,
                                  size_t bytesForRequest,
                                  const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
                                  size_t pageNum,
                                  bool isDirty));

MOCK_METHOD9(freezeSize, bool(pdb::PDBLoggerPtr &myLogger,
                              int port,
                              const std::string address,
                              bool onErr,
                              size_t bytesForRequest,
                              const std::function<bool(pdb::Handle<pdb::StoFreezeRequestResult>)> &processResponse,
                              pdb::PDBSetPtr setPtr,
                              size_t pageNum,
                              size_t numBytes));

MOCK_METHOD8(pinPage, bool(pdb::PDBLoggerPtr &myLogger,
                           int port,
                           const std::string &address,
                           bool onErr,
                           size_t bytesForRequest,
                           const std::function<bool(pdb::Handle<pdb::StoPinPageResult>)> &processResponse,
                           const pdb::PDBSetPtr &setPtr,
                           size_t pageNum));

};


class MockRequestFactory {
public:

  // the mock get page request
  template <class RequestType, class ResponseType, class ReturnType>
  static pdb::PDBPageHandle heapRequest(pdb::PDBLoggerPtr &myLogger,
                                            int port,
                                            const std::string &address,
                                            pdb::PDBPageHandle onErr,
                                            size_t bytesForRequest,
                                            const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::StoGetPageResult>)> &processResponse,
                                            const std::string &setName,
                                            const std::string &dbName,
                                            uint64_t pageNum) {

    return _requestFactory->getPage(myLogger, port, address, onErr, bytesForRequest, processResponse, setName, dbName, pageNum);
  }

  // the mock anonymous page request
  template <class RequestType, class ResponseType, class ReturnType>
  static pdb::PDBPageHandle heapRequest(pdb::PDBLoggerPtr &myLogger,
                                        int port,
                                        const std::string &address,
                                        pdb::PDBPageHandle onErr,
                                        size_t bytesForRequest,
                                        const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::StoGetPageResult>)> &processResponse,
                                        size_t minSize) {

    return _requestFactory->getAnonPage(myLogger, port, address, onErr, bytesForRequest, processResponse, minSize);
  }

  // return anonymous page
  template <class RequestType, class ResponseType, class ReturnType>
  static bool heapRequest(pdb::PDBLoggerPtr &myLogger,
                          int port,
                          const std::string &address,
                          bool onErr,
                          size_t bytesForRequest,
                          const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
                          size_t pageNum,
                          bool isDirty) {

    return _requestFactory->returnAnonPage(myLogger, port, address, onErr, bytesForRequest, processResponse, pageNum, isDirty);
  }

  // return page
  template <class RequestType, class ResponseType, class ReturnType>
  static bool heapRequest(pdb::PDBLoggerPtr &myLogger,
                          int port,
                          const std::string &address,
                          bool onErr,
                          size_t bytesForRequest,
                          const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
                          const std::string &setName,
                          const std::string &dbName,
                          size_t pageNum,
                          bool isDirty) {

    return _requestFactory->returnPage(myLogger, port, address, onErr, bytesForRequest, processResponse, setName, dbName, pageNum, isDirty);
  }

  template <class RequestType, class ResponseType, class ReturnType>
  static bool heapRequest(pdb::PDBLoggerPtr &myLogger,
                          int port,
                          const std::string &address,
                          bool onErr,
                          size_t bytesForRequest,
                          const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
                          PDBSetPtr &set,
                          size_t pageNum,
                          bool isDirty) {

    return _requestFactory->unpinPage(myLogger, port, address, onErr, bytesForRequest, processResponse, set, pageNum, isDirty);
  }

  // freeze size
  template <class RequestType, class ResponseType, class ReturnType>
  static bool heapRequest(pdb::PDBLoggerPtr &myLogger,
                          int port,
                          const std::string &address,
                          bool onErr,
                          size_t bytesForRequest,
                          const std::function<bool(pdb::Handle<pdb::StoFreezeRequestResult>)> &processResponse,
                          pdb::PDBSetPtr &setPtr,
                          size_t pageNum,
                          size_t numBytes) {

    return _requestFactory->freezeSize(myLogger, port, address, onErr, bytesForRequest, processResponse, setPtr, pageNum, numBytes);
  }

  // pin page
  template <class RequestType, class ResponseType, class ReturnType>
  static bool heapRequest(pdb::PDBLoggerPtr &myLogger,
                          int port,
                          const std::string &address,
                          bool onErr,
                          size_t bytesForRequest,
                          const std::function<bool(pdb::Handle<pdb::StoPinPageResult>)> &processResponse,
                          const pdb::PDBSetPtr &setPtr,
                          size_t pageNum) {

    return _requestFactory->pinPage(myLogger, port, address, onErr, bytesForRequest, processResponse, setPtr, pageNum);
  }

  static shared_ptr<MockRequestFactoryImpl> _requestFactory;
};

shared_ptr<MockRequestFactoryImpl> MockRequestFactory::_requestFactory = nullptr;

TEST(StorageManagerBackendTest, Test1) {

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

  pdb::PDBStorageManagerBackEnd<MockRequestFactory> storageManager(sharedMemory);

  MockRequestFactory::_requestFactory = std::make_shared<MockRequestFactoryImpl>();

  MockServer server;
  ON_CALL(server, getConfiguration).WillByDefault(testing::Invoke(
      [&]() {
        return std::make_shared<pdb::NodeConfig>();
      }));

  EXPECT_CALL(server, getConfiguration).Times(testing::AtLeast(1));

  storageManager.recordServer(server);

  /// 1. Mock the anonymous pages request

  ON_CALL(*MockRequestFactory::_requestFactory, getAnonPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger,
          int port,
          const std::string &address,
          pdb::PDBPageHandle onErr,
          size_t bytesForRequest,
          const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::StoGetPageResult>)> &processResponse,
          size_t minSize) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        int64_t myPage = pages.find(pageSize) == pages.end() ? curPage++ : pages.find(pageSize)->second;

        // make the page
        pdb::Handle<pdb::StoGetPageResult> returnPageRequest = pdb::makeObject<pdb::StoGetPageResult>(myPage * pageSize, myPage, true, false, -1, pageSize, "", "");

        // mark it as pinned
        pinned[myPage] = true;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly once
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

  // it should call send object exactly once
  EXPECT_CALL(*MockRequestFactory::_requestFactory, unpinPage).Times(2);

  /// 3. Mock the freeze size

  ON_CALL(*MockRequestFactory::_requestFactory, freezeSize).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string address, bool onErr,
          size_t bytesForRequest, const std::function<bool(pdb::Handle<pdb::StoFreezeRequestResult>)> &processResponse,
          pdb::PDBSetPtr setPtr, size_t pageNum, size_t numBytes) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);

        // make the page
        pdb::Handle<pdb::StoFreezeRequestResult> returnPageRequest = pdb::makeObject<pdb::StoFreezeRequestResult>(true);

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
          size_t bytesForRequest, const std::function<bool(pdb::Handle<pdb::StoPinPageResult>)> &processResponse,
          const pdb::PDBSetPtr &setPtr, size_t pageNum) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);

        // make the page
        pdb::Handle<pdb::StoPinPageResult>
            returnPageRequest = pdb::makeObject<pdb::StoPinPageResult>(pageNum * pageSize, true);

        // mark it as unpinned
        pinned[pageNum] = true;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly once
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

  // it should call send object exactly once
  EXPECT_CALL(*MockRequestFactory::_requestFactory, returnAnonPage).Times(98);

  {

    // grab two pages
    pdb::PDBPageHandle page1 = storageManager.getPage();
    pdb::PDBPageHandle page2 = storageManager.getPage();

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
      pdb::PDBPageHandle page3 = storageManager.getPage();
      pdb::PDBPageHandle page4 = storageManager.getPage();
      pdb::PDBPageHandle page5 = storageManager.getPage();
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
  storageManager.parent = nullptr;
}


}