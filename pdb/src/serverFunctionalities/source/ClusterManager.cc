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

#include <ClusterManager.h>
#include <SimpleRequestResult.h>
#include <SimpleRequest.h>

#include "CatalogClient.h"
#include "ClusterManager.h"
#include "CluSyncRequest.h"
#include "SimpleRequestHandler.h"


pdb::ClusterManager::ClusterManager(std::string address, int32_t port) : address(std::move(address)), port(port) {

  // grab the system info
  MemoryInfo memoryInfo{};
  getMemoryInfo(&memoryInfo);

  // grab the number of cores
  numCores = getCPUCores();

  // grab the total memory on this machine
  totalMemory = memoryInfo.totalram;

  // create the logger
  logger = make_shared<pdb::PDBLogger>("clusterManager.log");
}

void pdb::ClusterManager::registerHandlers(pdb::PDBServer &forMe) {
  forMe.registerHandler(
      CluSyncRequest_TYPEID,
      make_shared<SimpleRequestHandler<CluSyncRequest>>(
      [&](Handle<CluSyncRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // lock the catalog server
        std::lock_guard<std::mutex> guard(serverMutex);

        // generate the node identifier
        std::string nodeIdentifier = (std::string) request->nodeIP + ":" + std::to_string(request->nodePort);

        // sync the catalog server on this node with the one on the one that is requesting it.
        std::string error;
        bool success = getFunctionality<pdb::CatalogClient>().syncWithNode(std::make_shared<PDBCatalogNode>(nodeIdentifier,
                                                                                                            request->nodeIP,
                                                                                                            request->nodePort,
                                                                                                            request->nodeType,
                                                                                                            request->nodeMemory,
                                                                                                            request->nodeNumCores), error);

        // return the result
        return make_pair(success, error);
      }));
}

bool pdb::ClusterManager::syncManager(const std::string &managerAddress, int managerPort, std::string &error) {

  return simpleRequest<CluSyncRequest, SimpleRequestResult, bool>(
      logger, managerPort, managerAddress, false, 1024,
      [&](Handle<SimpleRequestResult> result) {
        if (result != nullptr) {
          if (!result->getRes().first) {
            error = "ClusterManager : Could not sink with the manager";
            logger->error(error);
            return false;
          }
          return true;
        }

        error = "ClusterManager : Could not sink with the manager";
        return false;
      },
      address, port, "worker", totalMemory, numCores);
}
