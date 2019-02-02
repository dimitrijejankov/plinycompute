//
// Created by dimitrije on 2/1/19.
//

#include <gtest/gtest.h>
#include <PDBStorageManagerBackEnd.h>
#include <gmock/gmock.h>

class MockServer : public pdb::PDBServer {
public:

  MOCK_METHOD0(getConfiguration, pdb::NodeConfigPtr());

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

MOCK_METHOD10(returnPage, bool(pdb::PDBLoggerPtr &myLogger,
                               int port,
                               const std::string &address,
                               bool onErr,
                               size_t bytesForRequest,
                               const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
                               const std::string &setName,
                               const std::string &dbName,
                               size_t pageNum,
                               bool isDirty));

MOCK_METHOD9(freezeSize, bool(pdb::PDBLoggerPtr &myLogger,
                              int port,
                              const std::string address,
                              bool onErr,
                              size_t bytesForRequest,
                              const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
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

  // return regular page
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

  // freeze size
  template <class RequestType, class ResponseType, class ReturnType>
  static bool heapRequest(pdb::PDBLoggerPtr &myLogger,
                          int port,
                          const std::string &address,
                          bool onErr,
                          size_t bytesForRequest,
                          const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
                          const pdb::PDBSetPtr &setPtr,
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

TEST(StorageManagerFrontendTest, Test1) {

  const size_t numPages = 16;
  const size_t pageSize = 64;

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

  ON_CALL(*MockRequestFactory::_requestFactory, getPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string address, pdb::PDBPageHandle onErr, size_t bytesForRequest,
          const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::StoGetPageResult>)> &processResponse, std::string setName, std::string dbName, uint64_t pageNum) {

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        //
        pdb::Handle<pdb::StoGetPageResult> returnPageRequest = pdb::makeObject<pdb::StoGetPageResult>(pageNum * pageSize, pageNum, false, false, -1, pageSize, setName, dbName);

        std::cout << "bLA" << std::endl;

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly once
  EXPECT_CALL(*MockRequestFactory::_requestFactory, getPage).Times(1);


  auto page = storageManager.getPage(make_shared<pdb::PDBSet>("db1", "set1"), 1);


  // just to remove the mock object
  MockRequestFactory::_requestFactory = nullptr;
}