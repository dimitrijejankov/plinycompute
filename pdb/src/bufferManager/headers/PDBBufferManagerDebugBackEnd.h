#pragma once

#ifdef DEBUG_BUFFER_MANAGER

#include <HeapRequest.h>

#include "BufGetPageResult.h"
#include "SimpleRequestResult.h"
#include "BufFreezeRequestResult.h"
#include "BufPinPageResult.h"
#include "PDBPageHandle.h"
#include "PDBSharedMemory.h"
#include "PDBBufferManagerBackEnd.h"

namespace pdb {

// we have debugging enabled therefore we need to use both the debug version and the real version
class PDBBufferManagerDebugBackEnd;
using PDBBufferManagerBackEndPtr = std::shared_ptr<PDBBufferManagerInterface>;
using PDBBufferManagerBackEndImpl = PDBBufferManagerInterface;

class PDBBufferManagerDebugBackendFactory {

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

    // init the request
    Handle<RequestType> request = makeObject<RequestType>(set, pageNum);

    // log the get page
    instance->logGetPage(set, pageNum, request->currentID);

    // make a request
    return RequestFactory::heapRequest<RequestType, ResponseType, ReturnType>(myLogger,
                                                                              port,
                                                                              address,
                                                                              onErr,
                                                                              bytesForRequest,
                                                                              processResponse,
                                                                              request);
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

    // init the request
    Handle<RequestType> request = makeObject<RequestType>(minSize);

    // log the get page, we don't care about the page number since it will be linked to the requested page
    instance->logGetPage(minSize, 0, request->currentID);

    // make a request
    return RequestFactory::heapRequest<RequestType, ResponseType, ReturnType>(myLogger,
                                                                              port,
                                                                              address,
                                                                              onErr,
                                                                              bytesForRequest,
                                                                              processResponse,
                                                                              request);
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

    // init the request
    Handle<RequestType> request = makeObject<RequestType>(pageNum, isDirty);

    // log free anonymous page
    instance->logFreeAnonymousPage(pageNum, request->currentID);

    // make a request
    return RequestFactory::heapRequest<RequestType, ResponseType, ReturnType>(myLogger,
                                                                              port,
                                                                              address,
                                                                              onErr,
                                                                              bytesForRequest,
                                                                              processResponse,
                                                                              request);
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
    // init the request
    Handle<RequestType> request = makeObject<RequestType>(setName, dbName, pageNum, isDirty);

    // log the down to zero references
    instance->logDownToZeroReferences(std::make_shared<PDBSet>(dbName, setName), pageNum, request->currentID);

    // make a request
    return RequestFactory::heapRequest<RequestType, ResponseType, ReturnType>(myLogger,
                                                                              port,
                                                                              address,
                                                                              onErr,
                                                                              bytesForRequest,
                                                                              processResponse,
                                                                              request);
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

    // init the request
    Handle<RequestType> request = makeObject<RequestType>(set, pageNum, isDirty);

    // log the unpin
    instance->logUnpin(set, pageNum, request->currentID);

    // make a request
    return RequestFactory::heapRequest<RequestType, ResponseType, ReturnType>(myLogger,
                                                                              port,
                                                                              address,
                                                                              onErr,
                                                                              bytesForRequest,
                                                                              processResponse,
                                                                              request);
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

    // init the request
    Handle<RequestType> request = makeObject<RequestType>(setPtr, pageNum, numBytes);

    // log freeze size
    instance->logFreezeSize(setPtr, pageNum, numBytes, request->currentID);

    // make a request
    return RequestFactory::heapRequest<RequestType, ResponseType, ReturnType>(myLogger,
                                                                              port,
                                                                              address,
                                                                              onErr,
                                                                              bytesForRequest,
                                                                              processResponse,
                                                                              request);
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

    // init the request
    Handle<RequestType> request = makeObject<RequestType>(setPtr, pageNum);

    // log repin
    instance->logRepin(setPtr, pageNum, request->currentID);

    // make a request
    return RequestFactory::heapRequest<RequestType, ResponseType, ReturnType>(myLogger,
                                                                              port,
                                                                              address,
                                                                              onErr,
                                                                              bytesForRequest,
                                                                              processResponse,
                                                                              request);
  }

  static PDBBufferManagerInterface* instance;
};

class PDBBufferManagerDebugBackEnd : public PDBBufferManagerBackEnd<PDBBufferManagerDebugBackendFactory> {
public:

  explicit PDBBufferManagerDebugBackEnd(const PDBSharedMemory &sharedMemory,
                                        const std::string &storageLocIn);

  void logGetPage(const PDBSetPtr &whichSet, uint64_t i, uint64_t timestamp) override;
  void logGetPage(size_t minBytes, uint64_t pageNumber, uint64_t timestamp) override;
  void logFreezeSize(const PDBSetPtr &setPtr, size_t pageNum, size_t numBytes, uint64_t timestamp) override;
  void logUnpin(const PDBSetPtr &setPtr, size_t pageNum, uint64_t timestamp) override;
  void logRepin(const PDBSetPtr &setPtr, size_t pageNum, uint64_t timestamp) override;
  void logFreeAnonymousPage(uint64_t pageNumber, uint64_t timestamp) override;
  void logDownToZeroReferences(const PDBSetPtr &setPtr, size_t pageNum, uint64_t timestamp) override;
  void logClearSet(const PDBSetPtr &set, uint64_t timestamp) override;
  void logExpect(const Handle<BufForwardPageRequest> &result) override;

private:

  enum class BufferManagerOperationType {
    GET_PAGE,
    FREEZE,
    UNPIN,
    REPIN,
    FREE,
    CLEAR,
    EXPECT
  };

  struct traceHasher {
    std::size_t operator()(std::vector<size_t> const& c) const  {

      std::size_t seed = 0;
      for(const auto &value : c) {
        seed ^= value + 0x9e3779b9 + (seed<< 6) + (seed>>2);
      }

      return seed;
    }
  };

  void logOperation(uint64_t timestamp,
                    PDBBufferManagerDebugBackEnd::BufferManagerOperationType operation,
                    const string &dbName,
                    const string &setName,
                    uint64_t pageNumber,
                    uint64_t value);

  /**
   * The file we are going to write all the debug symbols for the stack trace
   */
  int debugSymbolTableFile = 0;

  /**
   * The file where we are going to write all the stack traces
   */
  int stackTracesTableFile = 0;

  /**
   * The set of stack traces
   */
  std::unordered_map<std::vector<size_t>, size_t, traceHasher> stackTraces;

  /**
   * The lock we made
   */
  std::mutex m;

};


}

#endif