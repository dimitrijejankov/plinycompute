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

#ifndef DISPATCHER_SERVER_CC
#define DISPATCHER_SERVER_CC

#include "PDBDistributedStorage.h"
#include <snappy.h>
#include <HeapRequestHandler.h>
#include <DisAddData.h>
#include <BufGetPageRequest.h>
#include <PDBBufferManagerInterface.h>
#include <PDBDispatchRandomPolicy.h>
#include <DisDispatchData.h>
#include "PDBCatalogClient.h"
#include <boost/filesystem/path.hpp>

#define MAX_CONCURRENT_REQUESTS 10

namespace pdb {

void PDBDistributedStorage::init() {

  // init the policy
  policy = std::make_shared<PDBDispatchRandomPolicy>();

  // init the class
  logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "logs").string(), "PDBDistributedStorage.log");
}

void PDBDistributedStorage::registerHandlers(PDBServer &forMe) {

forMe.registerHandler(
    StoGetNextPageRequest_TYPEID,
    make_shared<pdb::HeapRequestHandler<pdb::StoGetNextPageRequest>>(
        [&](Handle<pdb::StoGetNextPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

          return handleGetNextPage<PDBCommunicator, RequestFactory>(request, sendUsingMe);
        }));

forMe.registerHandler(
    DisAddData_TYPEID,
    make_shared<HeapRequestHandler<pdb::DisAddData>>(
        [&](Handle<pdb::DisAddData> request, PDBCommunicatorPtr sendUsingMe) {

          /// 0. Check if the set exists

          if(!getFunctionalityPtr<PDBCatalogClient>()->setExists(request->databaseName, request->setName)) {

            // make the error string
            std::string errMsg = "The set does not exist!";

            // log the error
            logger->error(errMsg);

            // skip the data part
            sendUsingMe->skipBytes(errMsg);

            // create an allocation block to hold the response
            const UseTemporaryAllocationBlock tempBlock{1024};
            Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

            // sends result to requester
            sendUsingMe->sendObject(response, errMsg);
            return make_pair(false, errMsg);
          }

          /// 1. Receive the bytes onto an anonymous page

          // grab the buffer manager
          auto bufferManager = getFunctionalityPtr<PDBBufferManagerInterface>();

          // figure out how large the compressed payload is
          size_t numBytes = sendUsingMe->getSizeOfNextObject();

          // check if it is larger than a page
          if(bufferManager->getMaxPageSize() < numBytes) {

            // make the error string
            std::string errMsg = "The compressed size is larger than the maximum page size";

            // log the error
            logger->error(errMsg);

            // skip the data part
            sendUsingMe->skipBytes(errMsg);

            // create an allocation block to hold the response
            const UseTemporaryAllocationBlock tempBlock{1024};
            Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

            // sends result to requester
            sendUsingMe->sendObject(response, errMsg);
            return make_pair(false, errMsg);
          }

          // grab a page to write this
          auto page = bufferManager->getPage(numBytes);

          // receive bytes
          std::string error;
          sendUsingMe->receiveBytes(page->getBytes(), error);

          // check the uncompressed size
          size_t uncompressedSize = 0;
          snappy::GetUncompressedLength((char*) page->getBytes(), numBytes, &uncompressedSize);

          // check the uncompressed size
          if(bufferManager->getMaxPageSize() < uncompressedSize) {

            // make the error string
            std::string errMsg = "The uncompressed size is larger than the maximum page size";

            // log the error
            logger->error(errMsg);

            // create an allocation block to hold the response
            const UseTemporaryAllocationBlock tempBlock{1024};
            Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

            // sends result to requester
            sendUsingMe->sendObject(response, errMsg);
            return make_pair(false, errMsg);
          }

          /// 2. Figure out on what node to forward the thing

          // grab all active nodes
          const auto nodes = getFunctionality<PDBCatalogClient>().getActiveWorkerNodes();

          // if we have no nodes
          if(nodes.empty()) {

            // make the error string
            std::string errMsg = "There are no nodes where we can dispatch the data to!";

            // log the error
            logger->error(errMsg);

            // create an allocation block to hold the response
            const UseTemporaryAllocationBlock tempBlock{1024};
            Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

            // sends result to requester
            sendUsingMe->sendObject(response, errMsg);
            return make_pair(false, errMsg);
          }

          // get the next node
          auto node = policy->getNextNode(request->databaseName, request->setName, nodes);

          // time to send the stuff
          auto ret = RequestFactory::bytesHeapRequest<DisDispatchData, SimpleRequestResult, bool>(
              logger, node->port, node->address, false, 1024,
              [&](Handle<SimpleRequestResult> result) {

                if (result != nullptr && result->getRes().first) {
                  return true;
                }

                logger->error("Error sending data: " + result->getRes().second);
                error = "Error sending data: " + result->getRes().second;

                return false;
              },
              (char*) page->getBytes(), numBytes, request->databaseName, request->setName, request->typeName, numBytes);


          // create an allocation block to hold the response
          const UseTemporaryAllocationBlock tempBlock{1024};
          Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(ret, error);

          // sends result to requester
          ret = sendUsingMe->sendObject(response, error) && ret;

          return make_pair(ret, error);
    }));


}


}

#endif
