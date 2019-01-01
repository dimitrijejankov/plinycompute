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

#include <gtest/gtest.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <PDBServer.h>
#include <GenericWork.h>
#include <NodeConfig.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <ClusterManager.h>
#include <CatalogServer.h>
#include <PDBStorageManagerFrontEnd.h>
#include <ShutDown.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;


auto getConfiguration() {

  // the configuration for this node
  auto config = make_shared<pdb::NodeConfig>();

  config->isManager = true;
  config->address = "localhost";
  config->port = 8108;
  config->managerAddress = "localhost";
  config->managerPort = 8108;
  config->sharedMemSize = 2048;
  config->pageSize = 64;
  config->numThreads = 20;
  config->rootDirectory = "/tmp/pdbRoot";
  config->maxRetries = 5;

  // create the root directory
  fs::path rootPath(config->rootDirectory);
  if(!fs::exists(rootPath) && !fs::create_directories(rootPath)) {
    std::cout << "Failed to create the root directory!\n";
  }

  // init other parameters
  config->ipcFile = fs::path(config->rootDirectory).append("/ipcFile").string();
  config->catalogFile = fs::path(config->rootDirectory).append("/catalog").string();
  config->maxConnections = 100;

  // return the config
  return config;
}


TEST(StorageManager, StorageManagerBackend1) {

  // grab the configuration for this test
  auto config = getConfiguration();

  // init the storage manager, this has to be done before the fork!
  auto storageManager = std::make_shared<pdb::PDBStorageManagerFrontEnd>(config);

  // fork this to split into a frontend and backend
  pid_t pid = fork();

  // check whether we are the frontend or the backend
  if(pid == 0) {

    // do backend setup
    pdb::PDBLoggerPtr logger = make_shared<pdb::PDBLogger>("manager.log");
    pdb::PDBServer backEnd(pdb::PDBServer::NodeType::BACKEND, config, logger);

    // add the functionaries
    backEnd.addFunctionality<pdb::PDBStorageManagerInterface>(storageManager->getBackEnd());

    // start the backend
    backEnd.startServer(make_shared<pdb::GenericWork>([&](PDBBuzzerPtr callerBuzzer) {

      // wait 30 seconds for the frontend to initialize
      sleep(30);

      const int numOfThreads = 20;
      int counter = 0;

      // create the buzzer
      PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, int& cnt) {
        cnt++;
        PDB_COUT << "counter = " << cnt << std::endl;
      });

      for (unsigned long i = 0; i < numOfThreads; i++) {

        // grab a worker
        pdb::PDBWorkerPtr myWorker = backEnd.getWorkerQueue()->getWorker();

        // create some work for it
        pdb::PDBWorkPtr myWork = make_shared<pdb::GenericWork>([&, i](PDBBuzzerPtr callerBuzzer) {

          auto page1 = backEnd.getFunctionality<pdb::PDBStorageManagerInterface>().getPage(std::make_shared<pdb::PDBSet>("set", "db"), 1);
          auto page11 = backEnd.getFunctionality<pdb::PDBStorageManagerInterface>().getPage(std::make_shared<pdb::PDBSet>("set", "db"), 1);

          // we got back the page check that we have access to the page again since we requested the page back
          EXPECT_NE(page1->getBytes(), nullptr);

          // write some character
          if(page1->getBytes() != nullptr) {
            ((char*)page1->getBytes())[i] = (char)('a' + i);
            ((char*)page1->getBytes())[i + numOfThreads] = (char)('z' - i);
          }

          callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
        });

        // execute the work
        myWorker->execute(myWork, tempBuzzer);
      }

      // wait until all the nodes are finished
      while (counter < numOfThreads) {
        tempBuzzer->wait();
      }

      // add the null character at the end
      auto page = backEnd.getFunctionality<pdb::PDBStorageManagerInterface>().getPage(std::make_shared<pdb::PDBSet>("set", "db"), 1);

      // must be not null
      EXPECT_NE(page->getBytes(), nullptr);

      // ok check now if everything got written just fine
      if(page->getBytes() != nullptr) {
        ((char*)page->getBytes())[2 * numOfThreads + 1] = '\0';
        EXPECT_EQ(strcmp("abcdefghijklmnopqrstzyxwvutsrqponmlkjihg", ((char*)page->getBytes())), 0);
      }

      // shutdown the cluster
      backEnd.shutdownCluster();

      // buzz that we are done
      callerBuzzer->buzz(PDBAlarm::WorkAllDone);
    }));
  }
  else {

    // I'm the frontend server
    pdb::PDBLoggerPtr logger = make_shared<pdb::PDBLogger>("manager.log");
    pdb::PDBServer frontEnd(pdb::PDBServer::NodeType::FRONTEND, config, logger);

    // add the functionaries
    frontEnd.addFunctionality<pdb::PDBStorageManagerInterface>(storageManager);

    frontEnd.addFunctionality(std::make_shared<pdb::ClusterManager>());
    frontEnd.addFunctionality(std::make_shared<pdb::CatalogServer>());
    frontEnd.addFunctionality(std::make_shared<pdb::CatalogClient>(config->port, config->address, logger));

    frontEnd.startServer(make_shared<pdb::GenericWork>([&](PDBBuzzerPtr callerBuzzer) {

      // sync me with the cluster
      std::string error;
      frontEnd.getFunctionality<pdb::ClusterManager>().syncCluster(error);

      // log that the server has started
      std::cout << "Distributed storage manager server started!\n";

      // buzz that we are done
      callerBuzzer->buzz(PDBAlarm::WorkAllDone);
    }));
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}