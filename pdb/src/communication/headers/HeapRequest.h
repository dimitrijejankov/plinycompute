/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#ifndef SIMPLE_REQUEST_H
#define SIMPLE_REQUEST_H

#include "PDBLogger.h"
#include <snappy.h>
#include <PDBCommunicator.h>
#include <PDBConnectionManager.h>
#include <functional>


namespace pdb {

class RequestFactory {
public:

  // This templated function makes it easy to write a simple network client that asks a request,
  // then gets a result.  See, for example, CatalogClient.cc for an example of how to use.
  template <class RequestType, class ResponseType, class ReturnType, class... RequestTypeParams>
  static ReturnType heapRequest(pdb::PDBConnectionManager &conMgr, int nodeID,
                                ReturnType onErr, size_t bytesForRequest,
                                std::function<ReturnType(pdb::Handle<ResponseType>)> processResponse,
                                RequestTypeParams&&... args);


  // This method a vector of data in addition to the object of RequestType to the particular node.
  template <class RequestType, class DataType, class ResponseType, class ReturnType, class... RequestTypeParams>
  static ReturnType dataHeapRequest(pdb::PDBConnectionManager &conMgr, int nodeID,
                                    ReturnType onErr, size_t bytesForRequest, std::function<ReturnType(pdb::Handle<ResponseType>)> processResponse,
                                    pdb::Handle<Vector<pdb::Handle<DataType>>> dataToSend, RequestTypeParams&&... args);

  //
  template <class RequestType, class DataType, class ResponseType, class ReturnType, class... RequestTypeParams>
  static ReturnType dataKeyHeapRequest(pdb::PDBConnectionManager &conMgr, int nodeID,
                                       ReturnType onErr, size_t bytesForRequest, std::function<ReturnType(pdb::Handle<ResponseType>)> processResponse,
                                       pdb::Handle<Vector<pdb::Handle<DataType>>> dataToSend, RequestTypeParams&&... args);

  // This method send raw bytes in addition to the object of RequestType to the particular node.
  template <class RequestType, class ResponseType, class ReturnType, class... RequestTypeParams>
  static ReturnType bytesHeapRequest(pdb::PDBConnectionManager &conMgr, int nodeID,
                                     ReturnType onErr,
                                     size_t bytesForRequest,
                                     std::function<ReturnType(pdb::Handle<ResponseType>)> processResponse,
                                     char* bytes,
                                     size_t numBytes,
                                     RequestTypeParams&&... args);

  // This method waits for a response from the communicator
  template <class ResponseType, class ReturnType, class... RequestTypeParams>
  static ReturnType waitHeapRequest(const pdb::PDBLoggerPtr& logger,
                                    const pdb::PDBCommunicatorPtr& communicatorPtr, ReturnType onErr,
                                    std::function<ReturnType(pdb::Handle<ResponseType>)> processResponse);

  // Wait to for the bytes to arrive through the communicator
  static int64_t waitForBytes(const pdb::PDBLoggerPtr& logger,
                              const pdb::PDBCommunicatorPtr& communicatorPtr,
                              char* buffer, size_t bufferSize, std::string error);
};

};



#endif

#include "HeapRequestTemplate.cc"
