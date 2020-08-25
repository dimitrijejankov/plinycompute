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

#ifndef SIMPLE_REQUEST_CC
#define SIMPLE_REQUEST_CC

#include <functional>
#include <string>

#include "InterfaceFunctions.h"
#include "UseTemporaryAllocationBlock.h"
#include "PDBCommunicator.h"
#include "PDBConnectionManager.h"

using std::function;
using std::string;

#ifndef MAX_RETRIES
#define MAX_RETRIES 5
#endif
#define BLOCK_HEADER_SIZE 20
namespace pdb {

template<class RequestType, class ResponseType, class ReturnType, class... RequestTypeParams>
ReturnType RequestFactory::heapRequest(pdb::PDBConnectionManager &conMgr, int port, const std::string& address,
                                       ReturnType onErr, size_t bytesForRequest,
                                       std::function<ReturnType(pdb::Handle<ResponseType>)> processResponse,
                                       RequestTypeParams&&... args) {

  // get the logger
  auto &myLogger = conMgr.getLogger();

  // try multiple times if we fail to connect
  int numRetries = 0;
  while (numRetries <= MAX_RETRIES) {

      // used for error handling
      string errMsg;
      bool success;

      // connect to the server
      auto temp = conMgr.connectTo(myLogger, port, address, errMsg);
      if (temp == nullptr) {

          // log the error
          myLogger->error(errMsg);
          myLogger->error("Can not connect to remote server with port=" + std::to_string(port) + " and address=" + address + ");");

          // retry
          numRetries++;
          continue;
      }

      // log that we are connected
      myLogger->info(std::string("Successfully connected to remote server with port=") + std::to_string(port) + std::string(" and address=") + address);

      // check if it is invalid
      if (bytesForRequest <= BLOCK_HEADER_SIZE) {

          // ok this is an unrecoverable error
          myLogger->error("Too small buffer size for processing simple request");
          return onErr;
      }

      // make a block to send the request
      const UseTemporaryAllocationBlock tempBlock{bytesForRequest};

      // make the request
      Handle<RequestType> request = makeObject<RequestType>(args...);

      // send the object
      if (!temp->sendObject(request, errMsg)) {

          // yeah something happened
          myLogger->error(errMsg);
          myLogger->error("Not able to send request to server.\n");

          // we are done here we do not recover from this error
          return onErr;
      }

      // log that the object is sent
      myLogger->info("Object sent.");

      // get the response and process it
      ReturnType finalResult;
      size_t objectSize = temp->getSizeOfNextObject();

      // check if we did get a response
      if (objectSize == 0) {

          // ok we did not that sucks log what happened
          myLogger->error("We did not get a response.\n");

          // we are done here we do not recover from this error
          return onErr;
      }

      // allocate the memory
      std::unique_ptr<char[]> memory(new char[objectSize]);
      if (memory == nullptr) {

          errMsg = "FATAL ERROR in heapRequest: Can't allocate memory";
          myLogger->error(errMsg);

          /// TODO this needs to be an exception or something
          // this is a fatal error we should not be running out of memory
          exit(-1);
      }

      {
          Handle<ResponseType> result =  temp->getNextObject<ResponseType> (memory.get(), success, errMsg);
          if (!success) {

              // log the error
              myLogger->error(errMsg);
              myLogger->error("heapRequest: not able to get next object over the wire.\n");

              // we are done here we do not recover from this error
              return onErr;
          }

          finalResult = processResponse(result);
      }
      return finalResult;
  }

  //
  return onErr;
}

template<class RequestType, class DataType, class ResponseType, class ReturnType, class... RequestTypeParams>
ReturnType RequestFactory::dataHeapRequest(pdb::PDBConnectionManager &conMgr, int port, const std::string &address,
                                           ReturnType onErr, size_t bytesForRequest, std::function<ReturnType(pdb::Handle<ResponseType>)> processResponse,
                                           pdb::Handle<Vector<pdb::Handle<DataType>>> dataToSend, RequestTypeParams&&... args) {

  // get the logger
  auto logger = conMgr.getLogger();

  // get the record
  auto* myRecord = (Record<Vector<Handle<Object>>>*) getRecord(dataToSend);

  auto maxCompressedSize = snappy::MaxCompressedLength(myRecord->numBytes());

  // allocate the bytes for the compressed record
  std::unique_ptr<char[]> compressedBytes(new char[maxCompressedSize]);

  // compress the record
  size_t compressedSize;
  snappy::RawCompress((char*) myRecord, myRecord->numBytes(), compressedBytes.get(), &compressedSize);

  // log what we are doing
  logger->info("size before compression is "  + std::to_string(myRecord->numBytes()) + " and size after compression is " + std::to_string(compressedSize));

  int retries = 0;
  while (retries <= MAX_RETRIES) {

      // used for error handling
      string errMsg;
      bool success;

      // connect to the server
      auto temp = conMgr.connectTo(logger, port, address, errMsg);
      if (temp == nullptr) {

          // log the error
          logger->error(errMsg);
          logger->error("Can not connect to remote server with port=" + std::to_string(port) + " and address=" + address + ");");

          // retry
          retries++;
          continue;
      }

      // build the request
      if (bytesForRequest < HEADER_SIZE) {

          // log the error
          logger->error("block size is too small");

          // we are done here
          return onErr;
      }

      const UseTemporaryAllocationBlock tempBlock{bytesForRequest};
      Handle<RequestType> request = makeObject<RequestType>(args...);
      if (!temp->sendObject(request, errMsg)) {

          // log the error
          logger->error(errMsg);
          logger->error("simpleSendDataRequest: not able to send request to server.\n");

          // we are done here
          return onErr;
      }

      // now, send the bytes
      if (!temp->sendBytes(compressedBytes.get(), compressedSize, errMsg)) {

          logger->error(errMsg);
          logger->error("simpleSendDataRequest: not able to send data to server.\n");

          // we are done here
          return onErr;
      }

      // get the response and process it
      size_t objectSize = temp->getSizeOfNextObject();
      if (objectSize == 0) {

          // log the error
          logger->error("heapRequest: not able to get next object size");

          // we are done here
          return onErr;
      }

      std::unique_ptr<char[]> memory(new char[objectSize]);
      if (memory == nullptr) {

          // log the error
          logger->error("can't allocate memory");

          /// TODO this needs to be an exception or something
          // this is a fatal error we should not be running out of memory
          exit(-1);
      }

      ReturnType finalResult;
      {
          Handle<ResponseType> result = temp->getNextObject<ResponseType>(memory.get(), success, errMsg);
          if (!success) {

              // log the error
              logger->error(errMsg);
              logger->error("heapRequest: not able to get next object over the wire.\n");

              // we are done here
              return onErr;
          }

          finalResult = processResponse(result);
      }

      return finalResult;
  }

  return onErr;
}

template<class RequestType, class DataType, class ResponseType, class ReturnType, class... RequestTypeParams>
ReturnType RequestFactory::dataKeyHeapRequest(pdb::PDBConnectionManager &conMgr, int port, const std::string &address,
                                              ReturnType onErr, size_t bytesForRequest, function<ReturnType(Handle<ResponseType>)> processResponse,
                                              Handle<Vector<Handle<DataType>>> dataToSend, RequestTypeParams&&... args) {

  // get the logger
  const auto& logger = conMgr.getLogger();

  // get the record
  auto* myRecord = (Record<Vector<Handle<DataType>>>*) getRecord(dataToSend);

  auto maxCompressedSize = snappy::MaxCompressedLength(myRecord->numBytes());

  // allocate the bytes for the compressed record
  std::unique_ptr<char[]> compressedBytes(new char[maxCompressedSize]);

  // compress the record
  size_t compressedSize;
  snappy::RawCompress((char*) myRecord, myRecord->numBytes(), compressedBytes.get(), &compressedSize);

  // log what we are doing
  logger->info("size before compression is "  + std::to_string(myRecord->numBytes()) + " and size after compression is " + std::to_string(compressedSize));

  int retries = 0;
  while (retries <= MAX_RETRIES) {

    // used for error handling
    string errMsg;
    bool success;

    // connect to the server
    auto temp = conMgr.connectTo(logger, port, address, errMsg);
    if (temp == nullptr) {

      // log the error
      logger->error(errMsg);
      logger->error("Can not connect to remote server with port=" + std::to_string(port) + " and address=" + address + ");");

      // retry
      retries++;
      continue;
    }

    // build the request
    if (bytesForRequest < HEADER_SIZE) {

      // log the error
      logger->error("block size is too small");

      // we are done here
      return onErr;
    }

    const UseTemporaryAllocationBlock tempBlock{bytesForRequest};
    Handle<RequestType> request = makeObject<RequestType>(args...);
    if (!temp->sendObject(request, errMsg)) {

      // log the error
      logger->error(errMsg);
      logger->error("simpleSendDataRequest: not able to send request to server.\n");

      // we are done here
      return onErr;
    }

    // now, send the bytes
    if (!temp->sendBytes(compressedBytes.get(), compressedSize, errMsg)) {

      logger->error(errMsg);
      logger->error("simpleSendDataRequest: not able to send data to server.\n");

      // we are done here
      return onErr;
    }

    {
      // the type of the key
      using Key = typename std::remove_reference<decltype(((DataType*) nullptr)->getKey())>::type;

      // the block where we store the keys
      const UseTemporaryAllocationBlock keyBlock{myRecord->numBytes()};

      // the vector we just sent
      Handle<Vector<Handle<DataType>>> data = myRecord->getRootObject();

      // make the vector
      Handle<Vector<Key>> tmp = pdb::makeObject<Vector<Key>>();

      for(int i = 0; i < data->size(); ++i) {
        (*tmp).push_back((*data)[i]->getKey());
      }

      // store the records
      auto keyRecord = getRecord(tmp);

      // compress the record
      snappy::RawCompress((char*) keyRecord, keyRecord->numBytes(), compressedBytes.get(), &compressedSize);

      // now, send the bytes
      if (!temp->sendBytes(compressedBytes.get(), compressedSize, errMsg)) {

        logger->error(errMsg);
        logger->error("simpleSendDataRequest: not able to send key data to server.\n");

        // we are done here
        return onErr;
      }
    }

    // get the response and process it
    size_t objectSize = temp->getSizeOfNextObject();
    if (objectSize == 0) {

      // log the error
      logger->error("heapRequest: not able to get next object size");

      // we are done here
      return onErr;
    }

    std::unique_ptr<char[]> memory(new char[objectSize]);
    if (memory == nullptr) {

      // log the error
      logger->error("can't allocate memory");

      /// TODO this needs to be an exception or something
      // this is a fatal error we should not be running out of memory
      exit(-1);
    }

    ReturnType finalResult;
    {
      Handle<ResponseType> result = temp->getNextObject<ResponseType>(memory.get(), success, errMsg);
      if (!success) {

      // log the error
      logger->error(errMsg);
      logger->error("heapRequest: not able to get next object over the wire.\n");

      // we are done here
      return onErr;
      }

      finalResult = processResponse(result);
    }

    return finalResult;
  }

  return onErr;
}


template <class RequestType, class ResponseType, class ReturnType, class... RequestTypeParams>
ReturnType RequestFactory::bytesHeapRequest(pdb::PDBConnectionManager &conMgr, int port, const std::string& address, ReturnType onErr,
                                            size_t bytesForRequest, function<ReturnType(Handle<ResponseType>)> processResponse,
                                            char* bytes, size_t numBytes, RequestTypeParams&&... args) {


  // returns the logger
  auto &logger = conMgr.getLogger();

  // try to connect, if we fail we retry
  int retries = 0;
  while (retries <= MAX_RETRIES) {

      // used for error handling
      string errMsg;
      bool success;

      // connect to the server
      auto temp = conMgr.connectTo(logger, port, address, errMsg);
      if (temp == nullptr) {

          // log the error
          logger->error(errMsg);
          logger->error("Can not connect to remote server with port=" + std::to_string(port) + " and address=" + address + ");");

          // retry
          retries++;
          continue;
      }

      // build the request
      if (bytesForRequest < HEADER_SIZE) {

          // log the error
          logger->error("block size is too small");

          // we are done here
          return onErr;
      }

      const UseTemporaryAllocationBlock tempBlock{bytesForRequest};
      Handle<RequestType> request = makeObject<RequestType>(args...);
      if (!temp->sendObject(request, errMsg)) {

          // log the error
          logger->error(errMsg);
          logger->error("simpleSendDataRequest: not able to send request to server.\n");

          // we are done here
          return onErr;
      }

      // now, send the bytes
      if (!temp->sendBytes(bytes, numBytes, errMsg)) {

          logger->error(errMsg);
          logger->error("simpleSendDataRequest: not able to send data to server.\n");

          // we are done here
          return onErr;
      }

      // get the response and process it
      size_t objectSize = temp->getSizeOfNextObject();
      if (objectSize == 0) {

          // log the error
          logger->error("heapRequest: not able to get next object size");

          // we are done here
          return onErr;
      }

      std::unique_ptr<char[]> memory(new char[objectSize]);
      if (memory == nullptr) {

          // log the error
          logger->error("can't allocate memory");

          /// TODO this needs to be an exception or something
          // this is a fatal error we should not be running out of memory
          exit(-1);
      }

      ReturnType finalResult;
      {
          Handle<ResponseType> result = temp->getNextObject<ResponseType>(memory.get(), success, errMsg);
          if (!success) {

              // log the error
              logger->error(errMsg);
              logger->error("heapRequest: not able to get next object over the wire.\n");

              // we are done here
              return onErr;
          }

          finalResult = processResponse(result);
      }

      return finalResult;
  }

  return onErr;
}

template<class ResponseType, class ReturnType, class... RequestTypeParams>
ReturnType pdb::RequestFactory::waitHeapRequest(const pdb::PDBLoggerPtr& logger,
                                                const PDBCommunicatorPtr& communicatorPtr,
                                                ReturnType onErr,
                                                function<ReturnType(Handle<ResponseType>)> processResponse) {



    // get the response and process it
    size_t objectSize = communicatorPtr->getSizeOfNextObject();
    if (objectSize == 0) {

        // log the error
        logger->error("heapRequest: not able to get next object size");

        // we are done here
        return onErr;
    }

    std::unique_ptr<char[]> memory(new char[objectSize]);
    if (memory == nullptr) {

        // log the error
        logger->error("can't allocate memory");

        /// TODO this needs to be an exception or something
        // this is a fatal error we should not be running out of memory
        exit(-1);
    }

    // used for error handling
    string errMsg;
    bool success;

    ReturnType finalResult;
    {
        Handle<ResponseType> result = communicatorPtr->getNextObject<ResponseType>(memory.get(), success, errMsg);
        if (!success) {

            // log the error
            logger->error(errMsg);
            logger->error("heapRequest: not able to get next object over the wire.\n");

            // we are done here
            return onErr;
        }

        finalResult = processResponse(result);
    }

    return finalResult;
}


}
#endif
