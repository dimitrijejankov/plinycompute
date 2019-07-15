#pragma once

#include <HeapRequest.h>

#include "BufGetPageResult.h"
#include "SimpleRequestResult.h"
#include "BufFreezeRequestResult.h"
#include "BufPinPageResult.h"
#include "PDBPageHandle.h"
#include "PDBSharedMemory.h"
#include "PDBBufferManagerBackEnd.h"

namespace pdb {

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

    // make a request
    return RequestFactory::heapRequest<RequestType, ResponseType, ReturnType>(myLogger,
                                                                              port,
                                                                              address,
                                                                              onErr,
                                                                              bytesForRequest,
                                                                              processResponse,
                                                                              request);
  }

};

class PDBBufferManagerDebugBackEnd : public PDBBufferManagerBackEnd<PDBBufferManagerDebugBackendFactory> {
public:

  explicit PDBBufferManagerDebugBackEnd(const PDBSharedMemory &sharedMemory);


};


}
