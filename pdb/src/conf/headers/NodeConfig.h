//
// Created by dimitrije on 10/10/18.
//

#ifndef PDB_NODECONFIG_H
#define PDB_NODECONFIG_H

#include <string>
#include <memory>

namespace pdb {

struct NodeConfig;
typedef std::shared_ptr<NodeConfig> NodeConfigPtr;

struct NodeConfig {

  // parameter values
  bool isManager;

  /**
   * The address of the node
   */
  std::string address;

  /**
   * The port of the node
   */
  int32_t port;

  /**
   * The ip address of the manager
   */
  std::string managerAddress;

  /**
   * The port of the manger
   */
  int32_t managerPort;

  /**
   * The size of the buffer manager
   */
  size_t sharedMemSize;

  /**
   * Number of threads the execution engine is going to use
   */
  int32_t numThreads;

  /**
   * The maximum number of connections the server has
   */
  int32_t maxConnections;

  /**
   * The root directory of the node
   */
  std::string rootDirectory;

  /**
   * File to open a connection to the backend
   */
  std::string ipcFile;

  /**
   * The catalog file
   */
  std::string catalogFile;
};

}



#endif //PDB_NODECONFIG_H
