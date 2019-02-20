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
#include <functional>


namespace pdb {

class RequestFactory {
public:

  /**
   * This templated function makes it easy to write a simple network client that asks a request,
   * then gets a result.  See, for example, CatalogClient.cc for an example of how to use.
   *
   * @tparam RequestType - the type of object to create to send over the wire
   * @tparam ResponseType - the type of object we expect to receive over the wire
   * @tparam ReturnType - the type we will return to the caller
   * @tparam RequestTypeParams - type of the params to use for the constructor to the object we send over the wire
   *
   * @param myLogger - The logger we write error messages toThe logger we write error messages to
   * @param port - the port to send the request to
   * @param address - the address to send the request to
   * @param onErr - the value to return if there is an error sending/receiving data
   * @param bytesForRequest - the number of bytes to give to the allocator used to build the request
   * @param processResponse - the function used to process the response to the request
   * @param args - the arguments to give to the constructor of the request
   * @return whatever is returned from processResponse or onErr in case of failure
   */
  template <class RequestType, class ResponseType, class ReturnType, class... RequestTypeParams>
  static ReturnType heapRequest(pdb::PDBLoggerPtr myLogger,
                                int port,
                                std::string address,
                                ReturnType onErr,
                                size_t bytesForRequest,
                                std::function<ReturnType(pdb::Handle<ResponseType>)> processResponse,
                                RequestTypeParams&&... args);


  /**
   * This is a similar templated std::function that sends two objects, in sequence and then asks for the
   * results.
   *
   * @tparam RequestType - the type of object to create to send over the wire
   * @tparam SecondRequestType - the second object to create and send over the wirte
   * @tparam ResponseType - the type of object we expect to receive over the wire
   * @tparam ReturnType - the type we will return to the caller
   *
   * @param logger - The logger we write error messages to
   * @param port - the port to send the request to
   * @param address - the address to send the request to
   * @param onErr - the value to return if there is an error sending/receiving data
   * @param bytesForRequest - the number of bytes to give to the allocator used to build the request
   * @param processResponse - the std::function used to process the response to the request
   * @param firstRequest - the first request to send over the wire
   * @param secondRequest - the second request to send over the wire
   * @return whatever is returned from processResponse or onErr in case of failure
   */
  template <class RequestType, class SecondRequestType, class ResponseType, class ReturnType>
  static ReturnType doubleHeapRequest(pdb::PDBLoggerPtr logger,
                                      int port,
                                      std::string address,
                                      ReturnType onErr,
                                      size_t bytesForRequest,
                                      std::function<ReturnType(pdb::Handle<ResponseType>)> processResponse,
                                      pdb::Handle<RequestType> &firstRequest,
                                      pdb::Handle<SecondRequestType> &secondRequest);


  /**
   * This method a vector of data in addition to the object of RequestType to the particular node.
   *
   * @tparam RequestType - the type of object to create to send over the wire
   * @tparam DataType - the type of data we want to send
   * @tparam ResponseType - the type of object we expect to receive over the wire
   * @tparam ReturnType - the type we will return to the caller
   * @tparam RequestTypeParams - type of the params to use for the constructor to the object we send over the wire
   *
   * @param myLogger - The logger we write error messages to
   * @param port - the port to send the request to
   * @param address - the address to send the request to
   * @param onErr - the value to return if there is an error sending/receiving data
   * @param bytesForRequest - the number of bytes to give to the allocator used to build the request
   * @param processResponse - the std::function used to process the response to the request
   * @param dataToSend - the vector of data we want to send
   * @param args - the arguments to give to the constructor of the request
   * @return whatever is returned from processResponse or onErr in case of failure
   */
  template <class RequestType, class DataType, class ResponseType, class ReturnType, class... RequestTypeParams>
  static ReturnType dataHeapRequest(pdb::PDBLoggerPtr myLogger, int port, const std::string &address,
                                    ReturnType onErr, size_t bytesForRequest, std::function<ReturnType(pdb::Handle<ResponseType>)> processResponse,
                                    pdb::Handle<Vector<pdb::Handle<DataType>>> dataToSend, RequestTypeParams&&... args);


  /**
   * This method send raw bytes in addition to the object of RequestType to the particular node.
   * @tparam RequestType - the type of object to create to send over the wire
   * @tparam ResponseType - the type of data we want to get
   * @tparam ReturnType - the type of object we expect to receive over the wire
   * @tparam RequestTypeParams - type of the params to use for the constructor to the object we send over the wire
   * @param myLogger - The logger we write error messages to
   * @param port - the port to send the request to
   * @param address - the address to send the request to
   * @param onErr - the value to return if there is an error sending/receiving data
   * @param bytesForRequest - the number of bytes to give to the allocator used to build the request
   * @param processResponse - the function used to process the response to the request
   * @param bytes - a pointer to the bytes we want to send
   * @param numBytes - the number of bytes we want to send
   * @param args - arguments for the object of RequestType
   * @return whatever is returned from processResponse or onErr in case of failure
   */
  template <class RequestType, class ResponseType, class ReturnType, class... RequestTypeParams>
  static ReturnType bytesHeapRequest(pdb::PDBLoggerPtr myLogger,
                                     int port,
                                     std::string address,
                                     ReturnType onErr,
                                     size_t bytesForRequest,
                                     std::function<ReturnType(pdb::Handle<ResponseType>)> processResponse,
                                     char* bytes,
                                     size_t numBytes,
                                     RequestTypeParams&&... args);

  /**
   * This method waits for a response from the communicator
   * @tparam ResponseType - the type of data we want to get
   * @tparam ReturnType - the return type of the function
   * @param logger - the logger
   * @param communicatorPtr - the communicator
   * @param onErr - what to return on error
   * @param processResponse - the function to process the response
   * @return whatever is returned from processResponse or onErr in case of failure
   */
  template <class ResponseType, class ReturnType, class... RequestTypeParams>
  static ReturnType waitHeapRequest(pdb::PDBLoggerPtr logger,
                                    pdb::PDBCommunicatorPtr communicatorPtr,
                                    ReturnType onErr,
                                    std::function<ReturnType(pdb::Handle<ResponseType>)> processResponse);

  /**
   * Wait to for the bytes to arrive through the communicator
   *
   * @param logger - the logger
   * @param communicatorPtr - the communicator
   * @param buffer - where we want to put the bytes
   * @param bufferSize - the size of the buffer
   * @param error - error message if any
   * @return -1 if we fail, num of bytes we got
   */
  static int64_t waitForBytes(pdb::PDBLoggerPtr logger,
                              pdb::PDBCommunicatorPtr communicatorPtr,
                              char* buffer, size_t bufferSize, std::string error);
};

};



#endif

#include "HeapRequestTemplate.cc"
