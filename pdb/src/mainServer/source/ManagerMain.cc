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
#ifndef MASTER_MAIN_CC
#define MASTER_MAIN_CC

#include <iostream>
#include <string>
#include <ClusterManager.h>
#include <boost/filesystem/path.hpp>

#include "PDBServer.h"
#include "CatalogServer.h"
#include "PDBCatalogClient.h"
#include "PangeaStorageServer.h"
#include "DistributedStorageManagerServer.h"
#include "DispatcherServer.h"
#include "QuerySchedulerServer.h"
#include "StorageAddDatabase.h"
#include "SharedEmployee.h"

int main(int argc, char* argv[]) {
    int port = 8108;
    std::string managerIp;
    std::string pemFile = "conf/pdb.key";
    bool pseudoClusterMode = false;
    double partitionToCoreRatio = 0.75;
    if (argc == 3) {
        managerIp = argv[1];
        port = atoi(argv[2]);
    } else if ((argc == 4) || (argc == 5) || (argc == 6)) {
        managerIp = argv[1];
        port = atoi(argv[2]);
        std::string isPseudoStr(argv[3]);
        if (isPseudoStr.compare(std::string("Y")) == 0) {
            pseudoClusterMode = true;
            std::cout << "Running in standalone cluster mode" << std::endl;
        }
        if ((argc == 5) || (argc == 6)) {
            pemFile = argv[4];
        }
        if (argc == 6) {
            partitionToCoreRatio = stod(argv[5]);
        }

    } else {
        std::cout << "[Usage] #managerIp #port #runPseudoClusterOnOneNode (Y for running a "
                     "pseudo-cluster on one node, N for running a real-cluster distributedly, and "
                     "default is N) #pemFile (by default is conf/pdb.key) #partitionToCoreRatio "
                     "(by default is 0.75)"
                  << std::endl;
        exit(-1);
    }

    // the configuration for this node
    auto config = make_shared<pdb::NodeConfig>();

    config->isManager = true;
    config->address = managerIp;
    config->port = port;
    config->managerAddress = managerIp;
    config->maxConnections = 100;
    config->managerPort = port;
    config->sharedMemSize = 0; // nopangea
    config->ipcFile = ""; // nobackend
    config->rootDirectory = "./pdbRoot_" + managerIp + "_" + std::to_string(port);

    // create the root directory
    boost::filesystem::path rootPath(config->rootDirectory);
    if(!boost::filesystem::exists(rootPath) && !boost::filesystem::create_directories(rootPath)) {
      std::cout << "Failed to create the root directory!\n";
    }

    config->ipcFile = boost::filesystem::path(config->rootDirectory).append("/ipcFile").string();
    config->catalogFile = boost::filesystem::path(config->rootDirectory).append("/catalog").string();

    std::cout << "Starting up a distributed storage manager server...\n";
    pdb::PDBLoggerPtr myLogger = make_shared<pdb::PDBLogger>("frontendLogFile.log");
    pdb::PDBServer frontEnd(pdb::PDBServer::NodeType::FRONTEND, config, myLogger);

    ConfigurationPtr conf = make_shared<Configuration>();

    frontEnd.addFunctionality(std::make_shared<pdb::CatalogServer>());
    frontEnd.addFunctionality(std::make_shared<pdb::PDBCatalogClient>(port, "localhost", myLogger));

    std::string errMsg = " ";
    int numNodes = 1;
    string line;
    string nodeName;
    string hostName;
    string serverListFile;
    int portValue = 8108;

    serverListFile = pseudoClusterMode ? "conf/serverlist.test" : "conf/serverlist";

    frontEnd.addFunctionality(std::make_shared<pdb::DistributedStorageManagerServer>(myLogger));
    frontEnd.addFunctionality(std::make_shared<pdb::DispatcherServer>());
    frontEnd.addFunctionality(std::make_shared<pdb::QuerySchedulerServer>(port, myLogger, conf, pseudoClusterMode, partitionToCoreRatio));
    frontEnd.addFunctionality(std::make_shared<pdb::ClusterManager>());
    frontEnd.startServer(make_shared<GenericWork>([&](PDBBuzzerPtr callerBuzzer) {

      // sync me with the cluster
      std::string error;
      frontEnd.getFunctionality<ClusterManager>().syncCluster(error);

      // start sending the heart beats
      frontEnd.getFunctionality<ClusterManager>().startHeartBeat();

      // log that the server has started
      std::cout << "Distributed storage manager server started!\n";

      // buzz that we are done
      callerBuzzer->buzz(PDBAlarm::WorkAllDone);
    }));
}

#endif
