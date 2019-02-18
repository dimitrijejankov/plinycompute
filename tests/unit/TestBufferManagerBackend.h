//
// Created by dimitrije on 2/3/19.
//

#ifndef PDB_TESTBUFFERMANAGERBACKENDSINGLETHREADED_H
#define PDB_TESTBUFFERMANAGERBACKENDSINGLETHREADED_H

#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace pdb {

class MockServer : public pdb::PDBServer {
public:

  MOCK_METHOD0(getConfiguration, pdb::NodeConfigPtr());

  // mark the tests for the backend
  FRIEND_TEST(BufferManagerBackendTest, Test1);
};

class MockRequestFactoryImpl {
public:

MOCK_METHOD8(getPage, pdb::PDBPageHandle(pdb::PDBLoggerPtr &myLogger,
                                         int port,
                                         const std::string address,
                                         pdb::PDBPageHandle onErr,
                                         size_t bytesForRequest,
                                         const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::BufGetPageResult>)> &processResponse,
                                         pdb::PDBSetPtr set,
                                         uint64_t pageNum));

MOCK_METHOD7(getAnonPage, pdb::PDBPageHandle(pdb::PDBLoggerPtr &myLogger,
                                             int port,
                                             const std::string &address,
                                             pdb::PDBPageHandle onErr,
                                             size_t bytesForRequest,
                                             const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::BufGetPageResult>)> &processResponse,
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
                              const std::function<bool(pdb::Handle<pdb::BufFreezeRequestResult>)> &processResponse,
                              pdb::PDBSetPtr setPtr,
                              size_t pageNum,
                              size_t numBytes));

MOCK_METHOD8(pinPage, bool(pdb::PDBLoggerPtr &myLogger,
                           int port,
                           const std::string &address,
                           bool onErr,
                           size_t bytesForRequest,
                           const std::function<bool(pdb::Handle<pdb::BufPinPageResult>)> &processResponse,
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
                                            const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::BufGetPageResult>)> &processResponse,
                                            const pdb::PDBSetPtr set,
                                            uint64_t pageNum) {

    return _requestFactory->getPage(myLogger, port, address, onErr, bytesForRequest, processResponse, set, pageNum);
  }

  // the mock anonymous page request
  template <class RequestType, class ResponseType, class ReturnType>
  static pdb::PDBPageHandle heapRequest(pdb::PDBLoggerPtr &myLogger,
                                        int port,
                                        const std::string &address,
                                        pdb::PDBPageHandle onErr,
                                        size_t bytesForRequest,
                                        const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::BufGetPageResult>)> &processResponse,
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
                          const std::function<bool(pdb::Handle<pdb::BufFreezeRequestResult>)> &processResponse,
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
                          const std::function<bool(pdb::Handle<pdb::BufPinPageResult>)> &processResponse,
                          const pdb::PDBSetPtr &setPtr,
                          size_t pageNum) {

    return _requestFactory->pinPage(myLogger, port, address, onErr, bytesForRequest, processResponse, setPtr, pageNum);
  }

  static shared_ptr<MockRequestFactoryImpl> _requestFactory;
};

shared_ptr<MockRequestFactoryImpl> MockRequestFactory::_requestFactory = nullptr;

}


#endif //PDB_TESTBUFFERMANAGERBACKENDSINGLETHREADED_H
