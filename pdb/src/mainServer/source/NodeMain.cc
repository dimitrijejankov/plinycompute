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
#include <csignal>
#include <iostream>
#include <PDBServer.h>
#include <GenericWork.h>
#include <NodeConfig.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <utility>
#include <ClusterManager.h>
#include <PDBDistributedStorage.h>
#include "CatalogServer.h"
#include <PDBBufferManagerImpl.h>
#include <PDBStorageManager.h>
#include <PDBComputationServer.h>
#include <ExecutionServer.h>
#include <CUDAMemMgr.h>
#include <CUDAStaticStorage.h>
#include <CUDADynamicStorage.h>
#include <CUDAStreamManager.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace pdb;

// this is where our server is gonna be
pdb::PDBServerPtr server = nullptr;

// the handler for terminating the server
static void sig_stop(int signum) {
  server->stop();
}

void setGPUContext(void** gpuContext, pdb::PDBBufferManagerInterfacePtr myMgr, bool isManager){
    if (isManager) return;
    auto* tmp = new CUDAContext(std::move(myMgr));
    *gpuContext = static_cast<void*>(tmp);
}

extern void* gpuContext;
int main(int argc, char *argv[]) {

  // create the program options
  po::options_description desc{"Options"};

  // remove the old configuration if present
  bool removeOld;

  // the configuration for this node
  auto config = make_shared<pdb::NodeConfig>();

  // specify the options
  desc.add_options()("help,h", "Help screen");
  desc.add_options()("isManager,m", po::bool_switch(&config->isManager), "Start manager");
  desc.add_options()("address,i", po::value<std::string>(&config->address)->default_value("localhost"), "IP of the node");
  desc.add_options()("port,p", po::value<int32_t>(&config->port)->default_value(8108), "Port of the node");
  desc.add_options()("label,l", po::value<std::string>(&config->nodeLabel), "The label we assign to the node.");
  desc.add_options()("managerAddress,d", po::value<std::string>(&config->managerAddress)->default_value("localhost"), "IP of the manager");
  desc.add_options()("managerPort,o", po::value<int32_t>(&config->managerPort)->default_value(8108), "Port of the manager");
  desc.add_options()("sharedMemSize,s", po::value<size_t>(&config->sharedMemSize)->default_value(20480), "The size of the shared memory (MB)");
  desc.add_options()("pageSize,e", po::value<size_t>(&config->pageSize)->default_value(1024 * 1024 * 1024), "The size of a page (bytes)");
  desc.add_options()("numThreads,t", po::value<int32_t>(&config->numThreads)->default_value(2), "The number of threads we want to use");
  desc.add_options()("rootDirectory,r", po::value<std::string>(&config->rootDirectory)->default_value("./pdbRoot"), "The root directory we want to use.");
  desc.add_options()("maxRetries", po::value<uint32_t>(&config->maxRetries)->default_value(5), "The maximum number of retries before we give up.");
  desc.add_options()("debugBufferManager", po::bool_switch(&config->debugBufferManager), "Whether we want to debug the buffer manager or not. (has to be compiled for that)");

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

  // init other parameters
  config->catalogFile = fs::path(config->rootDirectory).append("/catalog").string();
  config->maxConnections = 200;

  if(!fs::exists(rootPath) && !fs::create_directories(rootPath)) {
    std::cout << "Failed to create the root directory!\n";
  }

  // init the storage manager, this has to be done before the fork!
  std::shared_ptr<pdb::PDBBufferManagerImpl> bufferManager;
  if(!config->debugBufferManager) {
    bufferManager = std::make_shared<pdb::PDBBufferManagerImpl>(config);
  }
  else {

    // if we have compiled with the appropriate flag we can use the debug buffer manager otherwise quit
    #ifdef DEBUG_BUFFER_MANAGER
      bufferManager = std::make_shared<pdb::PDBBufferManagerImpl>(config);
    #else
      std::cerr << "In order to use the debugBufferManager you have to compile with the flag -DDEBUG_BUFFER_MANAGER";
      exit(0);
    #endif
  }

  // create the server
  pdb::PDBLoggerPtr logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(config->rootDirectory) / "logs").string(),
                                                         "manager.log");
  server = std::make_shared<pdb::PDBServer>(config, logger);

  // add the functionaries
  server->addFunctionality<pdb::PDBBufferManagerInterface>(bufferManager);
  server->addFunctionality(std::make_shared<pdb::ClusterManager>());
  server->addFunctionality(std::make_shared<pdb::CatalogServer>());
  server->addFunctionality(std::make_shared<pdb::PDBDistributedStorage>());
  server->addFunctionality(std::make_shared<pdb::PDBCatalogClient>(config->port, config->address, logger));
  server->addFunctionality(std::make_shared<pdb::PDBStorageManager>());

  // on the worker put and execution server
  if(!config->isManager) {
    server->addFunctionality(std::make_shared<pdb::ExecutionServer>());
  }
  else {
    server->addFunctionality(std::make_shared<pdb::PDBComputationServer>());
  }

  // set the handler for shutdown
  struct sigaction action{};
  memset(&action, 0, sizeof(action));
  action.sa_handler = sig_stop;
  sigaction(SIGTERM, &action, nullptr);

  setGPUContext(&gpuContext, server->getFunctionalityPtr<PDBBufferManagerInterface>(), config->isManager);

  // start the server
  server->startServer(make_shared<pdb::GenericWork>([&](const PDBBuzzerPtr& callerBuzzer) {

    // sync me with the cluster
    std::string error;
    server->getFunctionality<pdb::ClusterManager>().syncCluster(error);

    // log that the server has started
    std::cout << "Distributed storage manager server started!\n";

    // buzz that we are done
    callerBuzzer->buzz(PDBAlarm::WorkAllDone);
  }));

  return 0;
}
