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

#include <boost/program_options.hpp>
#include <iostream>
#include <PDBServer.h>
#include <GenericWork.h>
#include <NodeConfig.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <ClusterManager.h>
#include <PDBDispatcherServer.h>
#include <CatalogServer.h>
#include <PDBBufferManagerFrontEnd.h>
#include <random>

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace pdb;

void writeBytes(int fileName, int pageNum, int pageSize, char *toMe) {

  char foo[1000];
  int num = 0;
  while (num < 900)
    num += sprintf(foo + num, "F: %d, P: %d ", fileName, pageNum);
  memcpy(toMe, foo, pageSize);
  sprintf(toMe + pageSize - 5, "END#");
}

pdb::PDBPageHandle createRandomPage(pdb::PDBBufferManagerInterface &myMgr, vector<pdb::PDBSetPtr> &mySets, vector<unsigned> &myEnds, vector<vector<size_t>> &lens) {

  // choose a set
  auto whichSet = lrand48() % mySets.size();

  // choose a length
  size_t len = 16;
  for (; (lrand48() % 3 != 0) && (len < 64); len *= 2);

  // store the random len
  lens[whichSet].push_back(len);

  pdb::PDBPageHandle returnVal = myMgr.getPage(mySets[whichSet], myEnds[whichSet]);
  writeBytes(whichSet, myEnds[whichSet], len, (char *) returnVal->getBytes());
  myEnds[whichSet]++;
  returnVal->freezeSize(len);
  return returnVal;
}

static int counter = 0;
pdb::PDBPageHandle createRandomTempPage(pdb::PDBBufferManagerImpl &myMgr, vector<size_t> &lengths) {

  // choose a length
  size_t len = 16;
  for (; (lrand48() % 3 != 0) && (len < 64); len *= 2);

  // store the length
  lengths.push_back(len);

  pdb::PDBPageHandle returnVal = myMgr.getPage();
  writeBytes(-1, counter, len, (char *) returnVal->getBytes());
  counter++;
  returnVal->freezeSize(len);
  return returnVal;
}


int main(int argc, char *argv[]) {

  // create the program options
  po::options_description desc{"Options"};

  // the configuration for this node
  auto config = make_shared<pdb::NodeConfig>();

  // specify the options
  desc.add_options()("help,h", "Help screen");
  desc.add_options()("isManager,m", po::bool_switch(&config->isManager), "Start manager");
  desc.add_options()("address,i", po::value<std::string>(&config->address)->default_value("localhost"), "IP of the node");
  desc.add_options()("port,p", po::value<int32_t>(&config->port)->default_value(8108), "Port of the node");
  desc.add_options()("managerAddress,d", po::value<std::string>(&config->managerAddress)->default_value("localhost"), "IP of the manager");
  desc.add_options()("managerPort,o", po::value<int32_t>(&config->managerPort)->default_value(8108), "Port of the manager");
  desc.add_options()("sharedMemSize,s", po::value<size_t>(&config->sharedMemSize)->default_value(2048), "The size of the shared memory (MB)");
  desc.add_options()("pageSize,e", po::value<size_t>(&config->pageSize)->default_value(1024 * 1024), "The size of a page (bytes)");
  desc.add_options()("numThreads,t", po::value<int32_t>(&config->numThreads)->default_value(1), "The number of threads we want to use");
  desc.add_options()("rootDirectory,r", po::value<std::string>(&config->rootDirectory)->default_value("./pdbRoot"), "The root directory we want to use.");
  desc.add_options()("maxRetries", po::value<uint32_t>(&config->maxRetries)->default_value(5), "The maximum number of retries before we give up.");

  // grab the options
  po::variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  // did somebody ask for help?
  if (vm.count("help")) {
    std::cout << desc << '\n';
    return 0;
  }

  // create the root directory
  fs::path rootPath(config->rootDirectory);
  if(!fs::exists(rootPath) && !fs::create_directories(rootPath)) {
    std::cout << "Failed to create the root directory!\n";
  }

  // init other parameters
  config->ipcFile = fs::path(config->rootDirectory).append("/ipcFile").string();
  config->catalogFile = fs::path(config->rootDirectory).append("/catalog").string();
  config->maxConnections = 100;

  // init the storage manager, this has to be done before the fork!
  auto storageManager = std::make_shared<pdb::PDBBufferManagerFrontEnd>(config);

  // fork this to split into a frontend and backend
  pid_t pid = fork();

  // check whether we are the frontend or the backend
  if(pid != 0) {

    // do backend setup
    pdb::PDBLoggerPtr logger = make_shared<pdb::PDBLogger>("manager.log");
    pdb::PDBServer backEnd(pdb::PDBServer::NodeType::BACKEND, config, logger);

    // add the functionaries
    backEnd.addFunctionality<pdb::PDBBufferManagerInterface>(storageManager->getBackEnd());

    // start the backend
    backEnd.startServer(make_shared<pdb::GenericWork>([&](PDBBuzzerPtr callerBuzzer) {

      // sync me with the cluster
      sleep(5);

      // log that the server has started
      std::cout << "Distributed storage manager server started!\n";

      // buzz that we are done
      callerBuzzer->buzz(PDBAlarm::WorkAllDone);
    }));
  }
  else {

    // I'm the frontend server
    pdb::PDBLoggerPtr logger = make_shared<pdb::PDBLogger>("manager.log");
    pdb::PDBServer frontEnd(pdb::PDBServer::NodeType::FRONTEND, config, logger);

    // add the functionaries
    frontEnd.addFunctionality<pdb::PDBBufferManagerInterface>(storageManager);

    frontEnd.addFunctionality(std::make_shared<pdb::ClusterManager>());
    frontEnd.addFunctionality(std::make_shared<pdb::CatalogServer>());
    frontEnd.addFunctionality(std::make_shared<pdb::PDBDispatcherServer>());
    frontEnd.addFunctionality(std::make_shared<pdb::PDBCatalogClient>(config->port, config->address, logger));

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

  return 0;
}
