//
// Created by dimitrije on 10/4/18.
//

#include <PDBHeartBeatWork.h>

#include "PDBHeartBeatWork.h"
#include "CatSetObjectTypeRequest.h"
#include "PDBCatalogClient.h"

PDBHeartBeatWork::PDBHeartBeatWork(pdb::PDBCatalogClient *client) : client(client), isStopped(false) {
  logger = make_shared<pdb::PDBLogger>("heartBeatLog.log");
}

void PDBHeartBeatWork::execute(PDBBuzzerPtr callerBuzzer) {

    while(!isStopped) {

        // sleep a while between rounds
        sleep(NODE_PING_DELAY * 10);

        // grab the worker nodes
        auto nodes = client->getWorkerNodes();

        // go through each node
        for(const auto &node : nodes) {

            // if the node is marked as inactive we don't check it, it needs to re-register with the cluster manager
            if(!node->active) {
              continue;
            }

            // send heartbeat
            std::cout << "Sending heart beat to node " << node->nodeID << std::endl;
            bool nodeStatus = sendHeartBeat(node->address, node->port);

            // update the status of the node
            std::string error;
            bool success = client->updateNodeStatus(node->nodeID, nodeStatus, error);

            // did we manage to update it
            if(!success) {
              logger->error("Could not update the status of the node with identifier : " + node->nodeID + "\n");
            }

            // sleep a while between individual pings.
            sleep(NODE_PING_DELAY);

            // in the case that we stopped between pinging of the nodes break
            if(isStopped) {
                break;
            }
        }
    }
}

void PDBHeartBeatWork::stop() {
    isStopped = true;
}

bool PDBHeartBeatWork::sendHeartBeat(const std::string &address, int32_t port) {

    return pdb::RequestFactory::heapRequest<pdb::CatSetObjectTypeRequest, pdb::SimpleRequestResult, bool>(
        logger, port, address, false, 1024,
        [&](pdb::Handle<pdb::SimpleRequestResult> result) {

          // did we get something back or not
          return result != nullptr;
        });
}
