//
// Created by dimitrije on 10/4/18.
//

#include <PDBHeartBeatWork.h>

#include "PDBHeartBeatWork.h"
#include "CatalogClient.h"

PDBHeartBeatWork::PDBHeartBeatWork(pdb::CatalogClient *client) : client(client), isStopped(false) {}

void PDBHeartBeatWork::execute(PDBBuzzerPtr callerBuzzer) {

    while(!isStopped) {

        // sleep a while between rounds
        sleep(NODE_PING_DELAY * 10);

        // grab the worker nodes
        auto nodes = client->getWorkerNodes();

        // go through each node
        for(const auto &node : nodes) {

            // send heartbeat
            std::cout << "Sending heart beat to node " << std::endl;

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
