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

#ifndef PDB_SERVER_CC
#define PDB_SERVER_CC

#include "BuiltInObjectTypeIDs.h"
#include "Handle.h"
#include "PDBAlarm.h"
#include <iostream>
#include <netinet/in.h>
#include "PDBServer.h"
#include "PDBWorker.h"
#include "ServerWork.h"
#include <signal.h>
#include <sys/socket.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
#include "PDBCommunicator.h"
#include "CloseConnection.h"
#include "PDBCatalogClient.h"
#include "ShutDown.h"
#include "ServerFunctionality.h"
#include "UseTemporaryAllocationBlock.h"
#include "SimpleRequestResult.h"
#include <memory>
#include <fstream>
#include <boost/filesystem/path.hpp>
#include "PDBConnectionManager.h"

namespace pdb {

PDBServer::PDBServer(const NodeConfigPtr &config, const PDBLoggerPtr &logger)
    : config(config), logger(logger), allDone(false), startedAcceptingRequests(false) {

  // ignore SIGPIPE
  struct sigaction sa{};
  memset(&sa, '\0', sizeof(sa));
  sa.sa_handler = SIG_IGN;
  sigaction(SIGPIPE, &sa, nullptr);

  // init the worker threads of this server
  workers = make_shared<PDBWorkerQueue>(logger, config->maxConnections);

  // init the connection manager
  connectionManager = std::make_shared<PDBConnectionManager>(config, logger);
  connectionManager->registerManager(config->managerAddress, config->managerPort);
}

void PDBServer::registerHandler(int16_t requestID, const PDBCommWorkPtr &handledBy) {
  handlers[requestID] = handledBy;
}

// this is the entry point for the listener to the port
void *callListenTCP(void *serverInstance) {
  auto *temp = static_cast<PDBServer *>(serverInstance);
  temp->listenTCP();
  return nullptr;
}

void PDBServer::listenTCP() {

  // initialize the connection manager
  connectionManager->init();

  // at this point we can say that we started accepting requests
  this->startedAcceptingRequests = true;

  // wait for someone to try to connect
  std::string errMsg;
  while (!allDone) {

    // listen for connections
    PDBCommunicatorPtr myCommunicator = connectionManager->listen(errMsg);
    if (myCommunicator == nullptr) {
      logger->error("PDBServer: could not point to an internet socket: " + errMsg);
      continue;
    }

    // log the info
    logger->info(std::string("accepted the connection with sockFD=") + std::to_string(myCommunicator->getSocketFD()));
    PDB_COUT << "||||||||||||||||||||||||||||||||||" << std::endl;
    PDB_COUT << "accepted the connection with sockFD=" << myCommunicator->getSocketFD() << std::endl;

    // handle the request
    handleRequest(myCommunicator);
  }
}

PDBCommunicatorPtr PDBServer::waitForConnection(const pdb::Handle<SerConnectToRequest> &connectionID) {

  // wait till we have the connection
  std::unique_lock<std::mutex> lk(this->m);
  cv.wait(lk, [&]{ return pendingConnections.find(*connectionID) != pendingConnections.end(); });

  // check if the socket is close if it is return null
  auto it = pendingConnections.find(*connectionID);
  if(it->second->isSocketClosed()) {

    // remove and return null
    pendingConnections.erase(it);
    return nullptr;
  }

  // grab the connection and return it
  auto connection = it->second;
  pendingConnections.erase(it);

  std::cout << "Got connection for file : " << connection->getSocketFD() << '\n';

  return connection;
}

PDBCommunicatorPtr PDBServer::connectTo(int32_t connectToMe, const pdb::Handle<SerConnectToRequest> &connectionID) {

  // stuff to keep track of the connecting
  std::string error;
  int numRetries = 0;

  // try to connect a bunch of times
  auto comm = connectionManager->connectTo(logger, connectToMe, error);
  while (comm == nullptr) {

    // log the error
    logger->error(error);
    logger->error("Can not connect to remove server with ID "+ std::to_string(connectToMe) + ";");

    // retry
    numRetries++;
    if(numRetries < getConfiguration()->maxRetries) {
      comm = connectionManager->connectTo(logger, connectToMe, error);
      continue;
    }

    // finish here since we are out of retries
    return nullptr;
  }

  // build the request
  if (!comm->sendObject(connectionID, error)) {

    // log the error
    logger->error(error);
    logger->error("Can not establish a connection to node.\n");

    // finish we could not send the object
    return nullptr;
  }

  // return the communicator
  return comm;
}

// gets access to worker queue
PDBWorkerQueuePtr PDBServer::getWorkerQueue() {
  return this->workers;
}

// gets access to logger
PDBLoggerPtr PDBServer::getLogger() {
  return this->logger;
}

pdb::NodeConfigPtr PDBServer::getConfiguration() {
  return this->config;
}

int32_t PDBServer::getNodeID() {
  return this->config->nodeID;
}

void PDBServer::handleRequest(const PDBCommunicatorPtr &myCommunicator) {

  ServerWorkPtr tempWork{make_shared<ServerWork>()};
  tempWork->setGuts(myCommunicator, this);
  PDBWorkerPtr tempWorker = workers->getWorker();
  tempWorker->execute(tempWork, tempWork->getLinkedBuzzer());
}

// returns true while we need to keep going... false when this connection is done
bool PDBServer::handleOneRequest(const PDBBuzzerPtr& callerBuzzer, const PDBCommunicatorPtr& myCommunicator) {

  // figure out what type of message the client is sending us
  int16_t requestID = myCommunicator->getObjectTypeID();
  string info;
  bool success;

  // if there was a request to close the connection, just get outta here
  if (requestID == CloseConnection_TYPEID) {
    UseTemporaryAllocationBlock tempBlock{2048};
    Handle<CloseConnection> closeMsg =
        myCommunicator->getNextObject<CloseConnection>(success, info);
    if (!success) {
      logger->error("PDBServer: close connection request, but was an error: " + info);
    } else {
      logger->trace("PDBServer: close connection request");
    }
    return false;
  }

  if (requestID == NoMsg_TYPEID) {
    logger->trace("PDBServer: the other side closed the connection");
    return false;
  }

  if (requestID == SerConnectToRequest_TYPEID) {

    UseTemporaryAllocationBlock tempBlock{2048};
    logger->trace("PDBServer: accepted remote connection string it.");

    // get the connect request
    Handle<SerConnectToRequest> connectRequest = myCommunicator->getNextObject<SerConnectToRequest>(success, info);
    if (!success) {
      logger->error("PDBServer: close connection request, but was an error: " + info);
    } else {
      logger->trace("PDBServer: close connection request");
    }

    // update the pending connections
    {
      // lock the mutex
      std::unique_lock<std::mutex> lck(this->m);

      // store te connection
      pendingConnections[*connectRequest] = myCommunicator;

      std::cout << "Connection established from node " << connectRequest->nodeID << " for task " <<  connectRequest->taskID << " file id " << myCommunicator->getSocketFD()  << "\n";

      // notify that we got a new connection
      cv.notify_all();
    }

    // finish
    return true;
  }

  // if we are asked to shut down...
  if (requestID == ShutDown_TYPEID) {
    UseTemporaryAllocationBlock tempBlock{2048};

    Handle<ShutDown> closeMsg = myCommunicator->getNextObject<ShutDown>(success, info);
    if (!success) {
      logger->error("PDBServer: close connection request, but was an error: " + info);
    } else {
      logger->trace("PDBServer: close connection request");
    }

    // if this is the manager
    if(getConfiguration()->isManager) {

      // go through all the nodes we got
      auto nodes = getFunctionality<PDBCatalogClient>().getActiveWorkerNodes();
      std::string errMsg;
      for(const auto& node : nodes) {

        // send the request for shutdown, we don't really care if it succeeds or not, if it does not we will log it
        // would make sense to make this more robust
        RequestFactory::heapRequest<ShutDown, SimpleRequestResult, bool>(*connectionManager, node->nodeID, false, 1024,
          [&](const Handle<SimpleRequestResult>& result) {

            // do we have a result
            if(result == nullptr) {

              errMsg = "Error getting type name: got nothing back from catalog";
              return false;
            }

            // did we succeed
            if (!result->getRes().first) {

              errMsg = "Error shutting down worker " + std::to_string(node->nodeID) +  " with "  + result->getRes().second;
              logger->error(errMsg);

              return false;
            }

            // we succeeded
            return true;
          });
      }
    }

    // ack the result
    std::string errMsg;
    Handle<SimpleRequestResult> result = makeObject<SimpleRequestResult>(true, "successful shutdown of server");
    if (!myCommunicator->sendObject(result, errMsg)) {
      logger->error("PDBServer: close connection request, but count not send response: " + errMsg);
    }

    // let everyone know we are done
    allDone = true;

    // stop listening
    connectionManager->stopListening();

    // mark that we have handled the request
    return true;
  }

  // and get a worker plus the appropriate work to service it
  if (handlers.count(requestID) == 0) {

    // there is not one, so send back an appropriate message
    logger->error("PDBServer: could not find an appropriate handler");
    return false;

    // in this case, got a handler
  } else {

    // End code replacement for testing

    // Chris' old code: (Observed problem: sometimes, buzzer never get buzzed.)
    // get a worker to run the handler (this blocks if no workers available)
    PDBWorkerPtr tempWorker = workers->getWorker();
    logger->trace("PDBServer: got a worker, start to do something...");
    logger->trace("PDBServer: requestID " + std::to_string(requestID));

    PDBCommWorkPtr tempWork = handlers[requestID]->clone();

    logger->trace("PDBServer: setting guts");
    tempWork->setGuts(myCommunicator, this);
    tempWorker->execute(tempWork, callerBuzzer);
    callerBuzzer->wait();
    logger->trace("PDBServer: handler has completed its work");
    return true;
  }
}

void PDBServer::signal(PDBAlarm signalWithMe) {
  workers->notifyAllWorkers(signalWithMe);
}

void PDBServer::startServer(const PDBWorkPtr& runMeAtStart) {

  // ignore broken pipe signals
  ::signal(SIGPIPE, SIG_IGN);

  // listen to the tcp socket
  int return_code = pthread_create(&externalListenerThread, nullptr, callListenTCP, this);
  if (return_code) {
    logger->error("ERROR; return code from pthread_create () is " + to_string(return_code));
    exit(-1);
  }

  // wait until the server starts listening to requests
  std::cout << "Waiting for the server to start accepting requests";
  while (this->startedAcceptingRequests != true) {
    std::cout << ".";
    usleep(300);
  }
  std::cout << "\n";

  // if there was some work to execute to start things up, then do it
  if (runMeAtStart != nullptr) {
    PDBBuzzerPtr buzzMeWhenDone = runMeAtStart->getLinkedBuzzer();
    PDBWorkerPtr tempWorker = workers->getWorker();
    tempWorker->execute(runMeAtStart, buzzMeWhenDone);
    buzzMeWhenDone->wait();
  }

  // wait for the thread to finish
  pthread_join(externalListenerThread, nullptr);

  // for each functionality, invoke its clean() method
  for (auto &functionality : functionalities) {
    functionality.second->cleanup();
  }

  // write the configuration to disk
  std::filebuf fb;
  boost::filesystem::path rootPath(config->rootDirectory);
  fb.open (rootPath / "config.conf", std::ios::out);
  std::ostream os(&fb);

  // write it out
  os << *config;

  // the shutdown
  std::cout << "Shutdown Cleanly!\n";
}

void PDBServer::stop() {
  allDone = true;
}

bool PDBServer::shutdownCluster() {

  // the copy we are about to make will be stored here
  const pdb::UseTemporaryAllocationBlock block(1024 * 1024);

  // create a communicator
  string errMsg;
  pdb::PDBCommunicatorPtr communicator = connectionManager->connectTo(logger, config->nodeID, errMsg);

  // did we fail to connect to the server
  if(communicator == nullptr) {
    return false;
  }

  // send the shutdown request to the manager
  pdb::Handle<pdb::ShutDown> collectStatsMsg = pdb::makeObject<pdb::ShutDown>();
  bool success = communicator->sendObject<pdb::ShutDown>(collectStatsMsg, errMsg);

  // we failed to send it
  if(!success) {
    return false;
  }

  // grab the response
  Handle<SimpleRequestResult> result = communicator->getNextObject<SimpleRequestResult>(success, errMsg);

  // if the result, is not null return the indicator
  if(result != nullptr) {
    return result->getRes().first;
  }

  // ok so result is null something went wrong
  return false;
}

}

#endif
