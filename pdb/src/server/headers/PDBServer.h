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
/*
 * File:   PDBServer.h
 * Author: Chris
 *
 * Created on September 25, 2015, 5:04 PM
 */

#ifndef PDBSERVER_H
#define PDBSERVER_H

#include <memory>
namespace pdb {

// create a smart pointer for PDBServer objects
class PDBServer;
typedef std::shared_ptr<PDBServer> PDBServerPtr;
}

#include "PDBAlarm.h"
#include "PDBCommWork.h"
#include "PDBLogger.h"
#include "PDBWork.h"
#include "PDBCommunicator.h"
#include "NodeConfig.h"
#include <string>
#include <map>
#include <atomic>

// This class encapsulates a multi-threaded sever in PDB.  The way it works is that one simply
// registers
// an event handler (encapsulated inside of a PDBWorkPtr); whenever there is a new connection that
// comes
// over the given port (or file in the case of a local socket) a PWBWorker is asked to handle the
// connection using the appropriate port using a cloned version of the specified PDBWork object.
//

namespace pdb {

class ServerFunctionality;

class PDBServer {
public:

  enum NodeType {
    FRONTEND,
    BACKEND
  };

  PDBServer() = default;

  PDBServer(NodeType type, const NodeConfigPtr &config, const PDBLoggerPtr &logger);

  // a server has many possible functionalities... storage, catalog client, query planning, etc.
  // to create and add a functionality, call this.  The Functionality class must derive from the
  // ServerFunctionality class, which means that it must implement the pure virtual function
  // RegisterHandlers (PDBServer &) that registers any special handlers that the class needs in
  // order to perform its required tasks, this method adds one server functionality
  template<class Functionality>
  void addFunctionality(std::shared_ptr<Functionality> functionality);

  // gets access to a particular functionality... this might be called (for example)
  template<class Functionality>
  Functionality &getFunctionality();

  // gets access to a particular functionality as a shared ptr... this might be called (for example)
  template<class Functionality>
  std::shared_ptr<Functionality> getFunctionalityPtr();

  // asks the server to handle a particular request coming over the wire with the particular work
  // type
  void registerHandler(int16_t typeID, const PDBCommWorkPtr &handledBy);

  // like registerHandler but repeat the work in a time interval
  // TODO: to be implemented later.
  //  void registerTimedHandler (uint32_t intervalInMilliseconds, PDBWorkPtr handledBy);

  // starts the server---this creates all of the threads and lets the server start taking
  // requests; this
  // call will never return.  Note that if runMeAtStart is not null, then runMeAtStart is executed
  // before the server starts handling requests
  void startServer(PDBWorkPtr runMeAtStart);

  // asks the server to signal all of the threads activily handling connections that a certain
  // event
  // has occured; this effectively just has us call PDBWorker.signal (signalWithMe) for all of the
  // workers that are currently handling requests.  Any that indicate that they have died as a
  // result
  // of the signal are forgotten (allowed to go out of scope) and then replaced with a new
  // PDBWorker
  // object
  void signal(PDBAlarm signalWithMe);

  // tell the server to start listening for people who want to connect
  void listen();

  // asks us to handle one request that is coming over the given PDBCommunicator; return true if
  // this
  // is not the last request over this PDBCommunicator object; buzzMeWhenDone is sent to the
  // worker that
  // is spawned to handle the request
  bool handleOneRequest(PDBBuzzerPtr buzzMeWhenDone, PDBCommunicatorPtr myCommunicator);

  void stop();  // added by Jia

  /**
   * Sends a request to the manager to shutdown the cluster
   * @return true if we succeed false otherwise
   */
  bool shutdownCluster();

  // Someone added this, but it is BAD!!  This should not be exposed
  // Jia: I understand it is bad, however we need to create threads in a handler, and I feel you
  // do
  //      not want multiple worker queue in one process. So I temprarily enabled this...
  virtual PDBWorkerQueuePtr getWorkerQueue();

  // gets access to logger
  virtual PDBLoggerPtr getLogger();

  // returns the configuration of this node
  virtual pdb::NodeConfigPtr getConfiguration();

private:

  // used to ask the most recently-added functionality to register its handlers
  void registerHandlersFromLastFunctionality();

  // is this frontend or a backend
  NodeType nodeType;

  // the configuration of this node
  pdb::NodeConfigPtr config;

  // when we get a message over the input socket, we'll handle it using the registered handler
  map<int16_t, PDBCommWorkPtr> handlers;

  // this is where all of our workers to handle the server requests live
  PDBWorkerQueuePtr workers;

  // handles a request using the given PDBCommunicator to obtain the data
  void handleRequest(const PDBCommunicatorPtr &myCommunicator);

  // true if we started accepting requests
  std::atomic_bool startedAcceptingRequests;

  // true when the server is done
  bool allDone;

  // where to log to
  PDBLoggerPtr logger;

  // used to run the server
  pthread_t listenerThread;

  // this is the socket we are listening to
  int sockFD;

  // this maps the name of a functionality class to a position
  std::map<std::string, size_t> functionalityNames;

  // this gives us each of the functionalities that the server can perform
  std::vector<shared_ptr<ServerFunctionality>> functionalities;
};
}

#include "ServerTemplates.cc"

#endif /* PDBSERVER_H */
