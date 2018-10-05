//
// Created by dimitrije on 10/4/18.
//

#include <PDBHeartBeatWork.h>

#include "PDBHeartBeatWork.h"
#include "CatSetObjectTypeRequest.h"
#include "CatalogClient.h"

PDBHeartBeatWork::PDBHeartBeatWork(pdb::CatalogClient *client) : client(client), isStopped(false) {
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

            // send heartbeat
            std::cout << "Sending heart beat to node " << node->nodeID << std::endl;
            sendHeartBeat(node->address, node->port);

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

    return pdb::simpleRequest<pdb::CatSetObjectTypeRequest, pdb::SimpleRequestResult, bool>(
        logger, port, address, false, 1024,
        [&](pdb::Handle<pdb::SimpleRequestResult> result) {

          // did we get something back or not
          return result != nullptr;
        });
}
