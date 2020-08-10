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
#include <HeapRequest.h>
#include <PDBHeartBeatWork.h>

#include "PDBCatalogClient.h"
#include "ClusterManager.h"
#include "CluSyncRequest.h"
#include "HeapRequestHandler.h"
#include "CluHeartBeatRequest.h"

namespace pdb {

// just map the stuff to get the system info to something reasonable
using MemoryInfo = struct sysinfo;
const auto &getMemoryInfo = sysinfo;
const auto &getCPUCores = std::thread::hardware_concurrency;

ClusterManager::ClusterManager() {

  // grab the system info
  MemoryInfo memoryInfo{};
  getMemoryInfo(&memoryInfo);

  // grab the number of cores
  numCores = getCPUCores();

  // grab the total memory on this machine
  totalMemory = memoryInfo.totalram / 1024;

  // create the logger
  logger = make_shared<PDBLogger>("clusterManager.log");
}

void ClusterManager::registerHandlers(PDBServer &forMe) {
  forMe.registerHandler(
      CluSyncRequest_TYPEID,
      make_shared<HeapRequestHandler<CluSyncRequest>>(
          [&](Handle<CluSyncRequest> request, PDBCommunicatorPtr sendUsingMe) {

            // lock the cluster manager
            std::lock_guard<std::mutex> guard(serverMutex);

            if (!getConfiguration()->isManager) {

              // create the response
              std::string error = "A worker node can not sync the cluster only a manager can!";
              Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, error);

              // sends result to requester
              bool success = sendUsingMe->sendObject(response, error);

              // return the result
              return make_pair(success, error);
            }

            // generate the node identifier
            std::string nodeIdentifier = (std::string) request->nodeIP + ":" + std::to_string(request->nodePort);

            // sync the catalog server on this node with the one on the one that is requesting it.
            std::string error;
            bool success = getFunctionality<PDBCatalogClient>().syncWithNode(std::make_shared<PDBCatalogNode>(nodeIdentifier,
                                                                                                           request->nodeIP,
                                                                                                           request->nodePort,
                                                                                                           request->nodeType,
                                                                                                           request->nodeNumCores,
                                                                                                           request->nodeMemory,
                                                                                                           true), error);

            // create an allocation block to hold the response
            const UseTemporaryAllocationBlock tempBlock{1024};

            // create the response
            Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(success, error);

            // sends result to requester
            success = sendUsingMe->sendObject(response, error) && success;

            // return the result
            return make_pair(success, error);
          }));

  forMe.registerHandler(
      CluHeartBeatRequest_TYPEID,
      make_shared<HeapRequestHandler<CluHeartBeatRequest>>(
          [&](Handle<CluHeartBeatRequest> request, PDBCommunicatorPtr sendUsingMe) {

            // create an allocation block to hold the response
            const UseTemporaryAllocationBlock tempBlock{1024};

            // create the response
            Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(true, "");

            // sends result to requester
            std::string error;
            bool success = sendUsingMe->sendObject(response, error);

            // return the result
            return make_pair(success, error);
          }));
}

bool ClusterManager::syncCluster(std::string &error) {

  // the configuration
  auto conf = getConfiguration();

  // figure out the type of the node
  std::string type = getConfiguration()->isManager ? "manager" : "worker";

  return RequestFactory::heapRequest< CluSyncRequest, SimpleRequestResult, bool>(
      logger, conf->managerPort, conf->managerAddress, false, 1024,
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
      conf->address, conf->port, type, totalMemory, numCores);
}

void ClusterManager::stopHeartBeat() {
    heartBeatWorker->stop();
}

void ClusterManager::startHeartBeat() {

    PDBWorkerPtr worker;
    // create a flush worker
    auto sender = make_shared<PDBHeartBeatWork>(&getFunctionality<PDBCatalogClient>());

    // find a thread in thread pool, if we can not find a thread, we block.
    while ((worker = this->getWorker()) == nullptr) {
        sched_yield();
    }

    // run the work
    worker->execute(sender, sender->getLinkedBuzzer());

    // set the worker
    heartBeatWorker = sender;
}

}
