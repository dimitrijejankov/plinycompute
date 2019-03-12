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

#ifndef PDB_PAGED_REQUEST_H
#define PDB_PAGED_REQUEST_H

#include "PDBBufferManagerInterface.h"
#include "PDBLogger.h"

namespace pdb {

#define BLOCK_HEADER_SIZE 20

/**
 * This templated function makes it easy to write a simple network client that asks a request,
 * then gets a result.
 * @tparam RequestType - this is the type of the object we are sending
 * @tparam ResponseType - the type of the response we are expecting from the other side
 * @tparam ReturnType - the return type
 * @tparam RequestTypeParams - the parameters of the request
 * @param logger - The logger we write error messages to
 * @param port - the port to send the request to
 * @param address - the address to send the request to
 * @param storage - the storage manager
 * @param onErr - the value to return if there is an error sending/receiving data
 * @param bytesForRequest - the number of bytes to give to the allocator used to build the request
 * @param processResponse - the function used to process the response to the request
 * @param args - the arguments to give to the constructor of the request
 * @return whatever the return ends up being
 */
template <class RequestType, class ResponseType, class ReturnType, class... RequestTypeParams>
ReturnType pagedRequest(const std::string &address, int port, const PDBBufferManagerInterfacePtr &storage, const PDBLoggerPtr &logger, ReturnType onErr,
                        const uint32_t maxRetries, size_t bytesForRequest, function<ReturnType(Handle<ResponseType>)> processResponse,
                        RequestTypeParams&&... args) {

  // then number of retries we do
  int numRetries = 0;

  // loop until we succeed or run out retries
  while (numRetries <= maxRetries) {

    PDBCommunicator temp;
    string errMsg;
    bool success = false;

    // connect to the internet server
    if (!temp.connectToInternetServer(logger, port, address, errMsg)) {

      // log the error
      logger->error(errMsg);
      logger->error("heapRequest: not able to connect to server.\n");

      // return onErr;
      std::cout << "ERROR: can not connect to remote server with port=" << port << " and address=" << address << std::endl;
      return onErr;
    }

    // is the block less then the minimum for a request
    if (bytesForRequest <= BLOCK_HEADER_SIZE) {
      std::cout << "ERROR: too small buffer size for processing simple request" << std::endl;
      return onErr;
    }

    // grab an anonymous page
    auto page = storage->getPage(bytesForRequest);
    const UseTemporaryAllocationBlock tempBlock{page->getBytes(), bytesForRequest};

    // make a request
    Handle<RequestType> request = makeObject<RequestType>(args...);
    if (!temp.sendObject(request, errMsg)) {

      // ok we failed log that
      logger->error(errMsg);
      logger->error("heapRequest: not able to send request to server.\n");

      // did we run out of retries?
      if (numRetries < maxRetries) {

        // no we haven't
        numRetries++;
        continue;

      } else {

        // return error
        return onErr;
      }
    }

    // get the response and process it
    ReturnType finalResult;
    size_t objectSize = temp.getSizeOfNextObject();
    if (objectSize == 0) {
      if (numRetries < maxRetries) {
        numRetries++;
        continue;
      } else {
        return onErr;
      }
    }

    page = storage->getPage(objectSize);
    {
      Handle<ResponseType> response = temp.getNextObject<ResponseType>(page->getBytes(), success, errMsg);
      if (!success) {

        logger->error(errMsg);
        logger->error("heapRequest: not able to get next object over the wire.\n");

        // did we run out of retries?
        if (numRetries < maxRetries) {

          // no we haven't
          numRetries++;
          continue;

        } else {

          // return error
          return onErr;
        }
      }

      // process the response and grab the response
      finalResult = processResponse(response);
    }

    // return the result
    return finalResult;
  }

  return onErr;
}

/**
 * This templated function makes it easy to write a simple network client that asks a request,
 * then gets a result.
 * @tparam RequestType - this is the type of the object we are sending
 * @tparam ResponseType - the type of the response we are expecting from the other side
 * @tparam ReturnType - the return type
 * @tparam RequestTypeParams - the parameters of the request
 * @param logger - The logger we write error messages to
 * @param communicator - communicator to where we want to send the message
 * @param storage - the storage manager
 * @param onErr - the value to return if there is an error sending/receiving data
 * @param bytesForRequest - the number of bytes to give to the allocator used to build the request
 * @param processResponse - the function used to process the response to the request
 * @param args - the arguments to give to the constructor of the request
 * @return whatever the return ends up being
 */
template <class RequestType, class ResponseType, class ReturnType, class... RequestTypeParams>
ReturnType pagedRequest(PDBCommunicatorPtr &communicator, const PDBBufferManagerInterfacePtr &storage, const PDBLoggerPtr &logger, ReturnType onErr,
                        const uint32_t maxRetries, size_t bytesForRequest, function<ReturnType(Handle<ResponseType>)> processResponse,
                        RequestTypeParams&&... args) {

  // then number of retries we do
  int numRetries = 0;

  // loop until we succeed or run out retries
  while (numRetries <= maxRetries) {

    string errMsg;
    bool success = false;

    // grab an anonymous page
    auto page = storage->getPage(bytesForRequest);
    const UseTemporaryAllocationBlock tempBlock{page->getBytes(), bytesForRequest};

    // make a request
    Handle<RequestType> request = makeObject<RequestType>(args...);
    if (!communicator->sendObject(request, errMsg)) {

      // ok we failed log that
      logger->error(errMsg);
      logger->error("heapRequest: not able to send request to server.\n");

      // did we run out of retries?
      if (numRetries < maxRetries) {

        // no we haven't
        numRetries++;
        continue;

      } else {

        // return error
        return onErr;
      }
    }

    // get the response and process it
    ReturnType finalResult;
    size_t objectSize = communicator->getSizeOfNextObject();
    if (objectSize == 0) {
      if (numRetries < maxRetries) {
        numRetries++;
        continue;
      } else {
        return onErr;
      }
    }

    page = storage->getPage(objectSize);
    {
      Handle<ResponseType> response = communicator->getNextObject<ResponseType>(page->getBytes(), success, errMsg);
      if (!success) {

        logger->error(errMsg);
        logger->error("heapRequest: not able to get next object over the wire.\n");

        // did we run out of retries?
        if (numRetries < maxRetries) {

          // no we haven't
          numRetries++;
          continue;

        } else {

          // return error
          return onErr;
        }
      }

      // process the response and grab the response
      finalResult = processResponse(response);
    }

    // return the result
    return finalResult;
  }

  return onErr;
}

}



#endif //PDB_PAGEDREQUEST_H
