#pragma once

#include <shared_mutex>
#include "PDBCommunicator.h"
#include "NodeConfig.h"

namespace pdb {

class PDBConnectionManager {
public:

  // makes a connection manager that can listen
  PDBConnectionManager(const NodeConfigPtr &config, PDBLoggerPtr logger);

  // make a connection manager that can listen
  explicit PDBConnectionManager(PDBLoggerPtr logger);

  // initializes the external socket
  bool init();

  // listens to the external socket, if it succeeds returns the communicator otherwise returns null
  PDBCommunicatorPtr listen(std::string &errMsg);

  // this connects to a server
  PDBCommunicatorPtr connectTo(const PDBLoggerPtr &logToMe, int portNumber,
                               const std::string &serverAddress, std::string &errMsg);

  // this connects to a server
  PDBCommunicatorPtr connectTo(const PDBLoggerPtr &logToMe, int nodeID, std::string &errMsg);

  // registers manager address
  void registerManager(const std::string &serverAddress, int portNumber);

  // register node
  void registerNode(int32_t nodeID, const std::string &serverAddress, int portNumber);

  // get the logger
  [[nodiscard]] const PDBLoggerPtr &getLogger() const;

  // returns the id of the manager
  int32_t getManagerID();

 private:

  // represents the address of a node
  struct NodeAddress {

    // the ip address of a node
    std::string ip;

    // the port of a node
    int32_t port;
  };

  // the nodes in the system
  std::map<uint32_t, NodeAddress> nodes;

  // mutex to lock the node information
  std::shared_mutex m;

  // listen to this port
  int32_t listenPort;

  // the number of retries
  int32_t maxRetries;

  // the socket to us
  int32_t externalSocket;

  // the logger
  PDBLoggerPtr logger;

  // the manager id
  const int32_t MANAGER_ID = -1;
};

//
using PDBConnectionManagerPtr = std::shared_ptr<PDBConnectionManager>;
}