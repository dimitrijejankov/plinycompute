#pragma once

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
  PDBCommunicatorPtr connectToInternetServer(const PDBLoggerPtr &logToMe, int portNumber,
                                             const std::string &serverAddress, std::string &errMsg);

  // get the logger
  const PDBLoggerPtr &getLogger() const;

 private:

  // listen to this port
  int32_t listenPort;

  // the number of retries
  int32_t maxRetries;

  // the socket to us
  int32_t externalSocket;

  // the logger
  PDBLoggerPtr logger;
};

//
using PDBConnectionManagerPtr = std::shared_ptr<PDBConnectionManager>;
}