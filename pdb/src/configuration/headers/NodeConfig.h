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
  bool isManager = false;

  /**
   * The address of the node
   */
  std::string address = "";

  /**
   * The port of the node
   */
  int32_t port = -1;

  /**
   * Whether we want to debug the buffer manager or not
   */
  bool debugBufferManager = false;

  /**
   * The ip address of the manager
   */
  std::string managerAddress = "";

  /**
   * The port of the manger
   */
  int32_t managerPort = -1;

  /**
   * The size of the buffer manager
   */
  size_t sharedMemSize = 0;

  /**
   * The size of the page
   */
  size_t pageSize = 0;

  /**
   * Number of threads the execution engine is going to use
   */
  int32_t numThreads = 0;

  /**
   * The maximum number of connections the server has
   */
  int32_t maxConnections = 0;

  /**
   * The maximum number of retries
   */
  uint32_t maxRetries = 0;

  /**
   * The root directory of the node
   */
  std::string rootDirectory = "";

  /**
   * File to open a connection to the backend
   */
  std::string ipcFile = "";

  /**
   * The catalog file
   */
  std::string catalogFile = "";

};

}



#endif //PDB_NODECONFIG_H
