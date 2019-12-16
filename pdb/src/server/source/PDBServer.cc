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
#include "ShutDown.h"
#include "ServerFunctionality.h"
#include "UseTemporaryAllocationBlock.h"
#include "SimpleRequestResult.h"
#include <memory>

namespace pdb {

PDBServer::PDBServer(NodeType type, const NodeConfigPtr &config, const PDBLoggerPtr &logger)
    : config(config), nodeType(type), logger(logger) {

  allDone = false;
  startedAcceptingRequests = 0;

  struct sigaction sa{};
  memset(&sa, '\0', sizeof(sa));
  sa.sa_handler = SIG_IGN;
  sigaction(SIGPIPE, &sa, nullptr);

  // init the worker threads of this server
  workers = make_shared<PDBWorkerQueue>(logger, config->maxConnections);
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

// this is the entry point for the listener to the ipc file
void *callListenIPC(void *serverInstance) {
  auto *temp = static_cast<PDBServer *>(serverInstance);
  temp->listenIPC();
  return nullptr;
}


void PDBServer::listenTCP() {

  // depending of whether we are are the frontend or backend we have different ports we need to listen to,
  int port;
  if (nodeType == NodeType::FRONTEND) {
    port = config->port;
  }
  else if(nodeType == NodeType::BACKEND) {
    port = config->backendPort;
  }
  else {
    throw runtime_error("Not sure what type of node this is, backend or frontend!");
  }

  // wait for an internet socket
  string errMsg;
  externalSocket = socket(AF_INET, SOCK_STREAM, 0);

  // added by Jia to avoid TimeWait state for old sockets
  int optval = 1;
  if (setsockopt(externalSocket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
    logger->error("PDBServer: couldn't setsockopt");
    logger->error(strerror(errno));
    std::cout << "PDBServer: couldn't setsockopt:" << strerror(errno) << std::endl;
    close(externalSocket);
    exit(0);
  }

  if (externalSocket < 0) {
    logger->error("PDBServer: could not get FD to internet socket");
    logger->error(strerror(errno));
    close(externalSocket);
    exit(0);
  }

  // bind the socket FD
  struct sockaddr_in serverAddress{};
  bzero((char *) &serverAddress, sizeof(serverAddress));
  serverAddress.sin_family = AF_INET;
  serverAddress.sin_addr.s_addr = INADDR_ANY;
  serverAddress.sin_port = htons((uint16_t) port);
  int retVal = ::bind(externalSocket, (struct sockaddr *) &serverAddress, sizeof(serverAddress));
  if (retVal < 0) {
    logger->error("PDBServer: could not bind to internet socket");
    logger->error(strerror(errno));
    close(externalSocket);
    exit(0);
  }

  logger->trace("PDBServer: about to listen to the Internet for a connection");

  // set the backlog on the socket
  if (::listen(externalSocket, 100) != 0) {
    logger->error("PDBServer: listen error");
    logger->error(strerror(errno));
    close(externalSocket);
    exit(0);
  }

  logger->trace("PDBServer: ready to go!");

  // at this point we can say that we started accepting requests
  this->startedAcceptingRequests += 1;

  // wait for someone to try to connect
  while (!allDone) {

    PDBCommunicatorPtr myCommunicator = make_shared<PDBCommunicator>();

    if (!myCommunicator->pointToInternet(logger, externalSocket, errMsg)) {
      logger->error("PDBServer: could not point to an internet socket: " + errMsg);
      continue;
    }
    logger->info(std::string("accepted the connection with sockFD=") +
        std::to_string(myCommunicator->getSocketFD()));
    PDB_COUT << "||||||||||||||||||||||||||||||||||" << std::endl;
    PDB_COUT << "accepted the connection with sockFD=" << myCommunicator->getSocketFD()
             << std::endl;
    handleRequest(myCommunicator);
  }
}

void PDBServer::listenIPC() {

  // depending of whether we are are the frontend or backend we have different ipc file we need to listen to.
  std::string ipcFile;
  if (nodeType == NodeType::FRONTEND) {
    ipcFile = config->backendIpcFile;
  }
  else if(nodeType == NodeType::BACKEND) {
    ipcFile = config->ipcFile;
  }

  // second, we are connecting to a local UNIX socket
  string errMsg;
  logger->trace("PDBServer: getting socket to file");
  internalSocket = socket(PF_UNIX, SOCK_STREAM, 0);

  if (internalSocket < 0) {
    logger->error("PDBServer: could not get FD to local socket");
    logger->error(strerror(errno));
    exit(0);
  }

  // bind the socket FD
  struct sockaddr_un serv_addr{};
  bzero((char *) &serv_addr, sizeof(serv_addr));
  serv_addr.sun_family = AF_UNIX;
  snprintf(serv_addr.sun_path, sizeof(serv_addr.sun_path), "%s", ipcFile.c_str());

  if (::bind(internalSocket, (struct sockaddr *) &serv_addr, sizeof(struct sockaddr_un))) {
    logger->error("PDBServer: could not bind to local socket");
    logger->error(strerror(errno));
    // if pathToBackEndServer exists, delete it.
    if (unlink(ipcFile.c_str()) == 0) {
      PDB_COUT << "Removed outdated " << ipcFile.c_str() << ".\n";
    }
    if (::bind(internalSocket, (struct sockaddr *) &serv_addr, sizeof(struct sockaddr_un))) {
      logger->error(
          "PDBServer: still could not bind to local socket after removing unixFile");
      logger->error(strerror(errno));
      exit(0);
    }
  }

  logger->debug("PDBServer: socket has name");
  logger->debug(serv_addr.sun_path);

  logger->trace("PDBServer: about to listen to the file for a connection");

  // set the backlog on the socket
  if (::listen(internalSocket, 100) != 0) {
    logger->error("PDBServer: listen error");
    logger->error(strerror(errno));
    exit(0);
  }

  logger->trace("PDBServer: ready to go!");

  // at this point we can say that we started accepting requests
  this->startedAcceptingRequests += 1;

  // wait for someone to try to connect
  while (!allDone) {

    PDBCommunicatorPtr myCommunicator;
    myCommunicator = make_shared<PDBCommunicator>();
    if (!myCommunicator->pointToFile(logger, internalSocket, errMsg)) {
      logger->error("PDBServer: could not point to an local UNIX socket: " + errMsg);
      continue;
    }
    PDB_COUT << "||||||||||||||||||||||||||||||||||" << std::endl;
    PDB_COUT << "accepted the connection with sockFD=" << myCommunicator->getSocketFD()
             << std::endl;
    handleRequest(myCommunicator);
  }
}

PDBCommunicatorPtr PDBServer::waitForConnection(const pdb::Handle<SerConnectToRequest> &connectionID) {

  // wait till we have the connection
  std::unique_lock<std::mutex> lk(m);
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

  return connection;
}

PDBCommunicatorPtr PDBServer::connectTo(const std::string &ipcFile, const pdb::Handle<SerConnectToRequest> &connectionID) {

  // stuff to keep track of the connecting
  std::string error;
  int numRetries = 0;

  // try to connect a bunch of times
  auto comm = std::make_shared<PDBCommunicator>();
  while (!comm->connectToLocalServer(logger, ipcFile, error)) {

    // log the error
    logger->error(error);
    logger->error("Can not connect to local server with ipc file " + ipcFile + ");");

    // retry
    numRetries++;
    if(numRetries < getConfiguration()->maxRetries) {
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

PDBCommunicatorPtr PDBServer::connectTo(const std::string &ip, int32_t port, const pdb::Handle<SerConnectToRequest> &connectionID) {

  // stuff to keep track of the connecting
  std::string error;
  int numRetries = 0;

  // try to connect a bunch of times
  auto comm = std::make_shared<PDBCommunicator>();
  while (!comm->connectToInternetServer(logger, port, ip, error)) {

    // log the error
    logger->error(error);
    logger->error("Can not connect to remove server with ip " + ip + " and port " + std::to_string(port) + ";");

    // retry
    numRetries++;
    if(numRetries < getConfiguration()->maxRetries) {
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

      std::cout << "Connection established from node " << connectRequest->nodeID << " for task " <<  connectRequest->taskID << "\n";

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

    // ack the result
    std::string errMsg;
    Handle<SimpleRequestResult> result = makeObject<SimpleRequestResult>(true, "successful shutdown of server");
    if (!myCommunicator->sendObject(result, errMsg)) {
      logger->error("PDBServer: close connection request, but count not send response: " + errMsg);
    }

    PDB_COUT << "Cleanup server functionalities" << std::endl;

    // for each functionality, invoke its clean() method
    for (auto &functionality : functionalities) {
      functionality->cleanup();
    }

    // kill the FD and let everyone know we are done
    allDone = true;

    // close(sockFD);
    // we can't simply close socket like this, because there are still incoming
    // messages in accepted connections
    // use reuse address option instead
    return false;
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

  // listen to the ipc socket
  return_code = pthread_create(&internalListenerThread, nullptr, callListenIPC, this);
  if (return_code) {
    logger->error("ERROR; return code from pthread_create () is " + to_string(return_code));
    exit(-1);
  }

  // wait until the server starts listening to requests
  std::cout << "Waiting for the server to start accepting requests";
  while (this->startedAcceptingRequests != 2) {
    std::cout << ".";
    usleep(300);
  }
  std::cout << "\n";

  // just to keep it safe
  usleep(300);

  // if there was some work to execute to start things up, then do it
  if (runMeAtStart != nullptr) {
    PDBBuzzerPtr buzzMeWhenDone = runMeAtStart->getLinkedBuzzer();
    PDBWorkerPtr tempWorker = workers->getWorker();
    tempWorker->execute(runMeAtStart, buzzMeWhenDone);
    buzzMeWhenDone->wait();
  }

  // and now just sleep
  while (!allDone) {
    sleep(1);
  }
}

void PDBServer::registerHandlersFromLastFunctionality() {
  functionalities[functionalities.size() - 1]->recordServer(*this);
  functionalities[functionalities.size() - 1]->init();
  functionalities[functionalities.size() - 1]->registerHandlers(*this);
}

void PDBServer::stop() {
  allDone = true;
}

bool PDBServer::shutdownCluster() {

  // the copy we are about to make will be stored here
  const pdb::UseTemporaryAllocationBlock block(1024 * 1024);

  // create a communicator
  pdb::PDBCommunicatorPtr communicator = std::make_shared<pdb::PDBCommunicator>();

  // try to connect to the node
  string errMsg;
  bool failure = communicator->connectToInternetServer(logger, config->managerPort, config->managerAddress, errMsg);

  // did we fail to connect to the server
  if(!failure) {
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
