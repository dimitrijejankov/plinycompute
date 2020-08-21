//
// Created by dimitrije on 10/10/18.
//

#ifndef PDB_NODECONFIG_H
#define PDB_NODECONFIG_H

#include <string>
#include <memory>
#include <ostream>
#include <boost/algorithm/string.hpp>

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
   * The id of the node
   */
  int32_t nodeID = -1;

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
   * Label of the node
   */
  std::string nodeLabel = "";

  /**
   * The root directory of the node
   */
  std::string rootDirectory = "";

  /**
   * The catalog file
   */
  std::string catalogFile = "";

  friend ostream &operator<<(ostream &os, const NodeConfig &config) {

    // serialize
    os << "isManager" << " \"" << config.isManager << "\"\n";
    os << "address" << " \"" << config.address << "\"\n";
    os << "port" << " \"" << config.port << "\"\n";
    os << "debugBufferManager" << " \"" << config.debugBufferManager << "\"\n";
    os << "managerAddress" << " \"" << config.managerAddress << "\"\n";
    os << "managerPort" << " \"" << config.managerPort << "\"\n";
    os << "sharedMemSize" << " \"" << config.sharedMemSize << "\"\n";
    os << "pageSize" << " \"" << config.pageSize << "\"\n";
    os << "numThreads" << " \"" << config.numThreads << "\"\n";
    os << "maxConnections" << " \"" << config.maxConnections << "\"\n";
    os << "maxRetries" << " \"" << config.maxRetries << "\"\n";
    os << "nodeLabel" << " \"" << config.nodeLabel << "\"\n";
    os << "rootDirectory" << " \"" << config.rootDirectory << "\"\n";
    os << "catalogFile" << " \"" << config.catalogFile << "\"\n";
    os << "nodeID" << " \"" << config.nodeID << "\"";

    return os;
  }

  friend istream &operator>>(istream &is, NodeConfig &config) {

    std::string label;
    std::string value;
    for (;;) {

      // read the label
      is >> label;
      std::getline(is, value);
      auto it = value.find_first_of('\"');
      auto jt = value.find_last_of('\"');

      // check if have parsed it correctly
      if(it == std::string::npos || jt == std::string::npos) {
        std::cerr << "Could not parse the configuration.\n";
        exit(-1);
      }

      // extract the value
      value[it] = ' ';
      value[jt] = ' ';
      boost::trim(value);

      // check what we need to parse
      if (label == "isManager") { std::stringstream(value) >> config.isManager; }
      else if (label == "address") { std::stringstream(value) >> config.address; }
      else if (label == "port") { std::stringstream(value) >> config.port; }
      else if (label == "debugBufferManager") { std::stringstream(value) >> config.debugBufferManager; }
      else if (label == "managerAddress") { std::stringstream(value) >> config.managerAddress; }
      else if (label == "managerPort") { std::stringstream(value) >> config.managerPort; }
      else if (label == "sharedMemSize") { std::stringstream(value) >> config.sharedMemSize; }
      else if (label == "pageSize") { std::stringstream(value) >> config.pageSize; }
      else if (label == "numThreads") { std::stringstream(value) >> config.numThreads; }
      else if (label == "maxConnections") { std::stringstream(value) >> config.maxConnections; }
      else if (label == "maxRetries") { std::stringstream(value) >> config.maxRetries; }
      else if (label == "nodeLabel") { std::stringstream(value) >> config.nodeLabel; }
      else if (label == "rootDirectory") { std::stringstream(value) >> config.rootDirectory; }
      else if (label == "catalogFile") { std::stringstream(value) >> config.catalogFile; }
      else if (label == "nodeID") { std::stringstream(value) >> config.nodeID; }

      if (is.eof()) break;
    }

    // return the stream
    return is;
  }
};

}

#endif //PDB_NODECONFIG_H
