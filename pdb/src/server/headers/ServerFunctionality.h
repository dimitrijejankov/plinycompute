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

#ifndef SERVER_FUNCT_H
#define SERVER_FUNCT_H

#include "PDBServer.h"

namespace pdb {

// this pure virtual class encapsulates some particular server functionality (catalog client,
// catalog server, storage server, etc.).
class ServerFunctionality {

 public:

  // registers any particular handlers that this server needs
  virtual void registerHandlers(PDBServer &forMe) = 0;

  // this method is called by the PDBServer before registering the handlers but after the functionality is created.
  virtual void init() {};

  // added by Jia, it will be invoked when PDBServer is to be shutdown
  virtual void cleanup() {}

  // access a particular functionality on the attached server
  template<class Functionality>
  Functionality &getFunctionality() {
    return parent->getFunctionality<Functionality>();
  }

  // access a particular functionality on the attached server as a shared pointer
  template<class Functionality>
  std::shared_ptr<Functionality> getFunctionalityPtr() {
    return std::move(parent->getFunctionalityPtr<Functionality>());
  }

  /**
   * Waits for somebody to connect to this server with the following connection id
   * @param connectionID
   * @return
   */
  PDBCommunicatorPtr waitForConnection(const pdb::Handle<SerConnectToRequest> &connectionID) {
    return this->parent->waitForConnection(connectionID);
  }

  /**
   * Connect to a particular server
   * @param ip - the ip of the node
   * @param port - port
   * @param connectionID - connection id
   * @return the communicator if we succeed, null otherwise
   */
  PDBCommunicatorPtr connectTo(const std::string &ip,
                               int32_t port,
                               const pdb::Handle<SerConnectToRequest> &connectionID) {
    return this->parent->connectTo(ip, port, connectionID);
  }

  /**
   * Connect to a particular server, through an ipcFile
   * @param ipcFile - the ipc file of the server
   * @param connectionID - connection id
   * @return the communicator if we succeed, null otherwise
   */
  PDBCommunicatorPtr connectTo(const std::string &ipcFile, const pdb::Handle<SerConnectToRequest> &connectionID) {
    return this->parent->connectTo(ipcFile, connectionID);
  }

  // remember the server this is attached to
  void recordServer(PDBServer &recordMe) {
    parent = &recordMe;
  }

  PDBWorkerQueuePtr getWorkerQueue() {
    return parent->getWorkerQueue();
  }

  PDBWorkerPtr getWorker() {
    return parent->getWorkerQueue()->getWorker();
  }

  PDBLoggerPtr getLogger() {
    return parent->getLogger();
  }

  NodeConfigPtr getConfiguration() {
    return parent->getConfiguration();
  }

  int32_t getNodeID() {
      return parent->getNodeID();
  };

 protected:
  PDBServer *parent = nullptr;

};
}

#endif
