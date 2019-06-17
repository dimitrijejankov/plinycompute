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
 * CatSyncRequest.h
 *
 */

#ifndef CATALOG_NODE_METADATA_H_
#define CATALOG_NODE_METADATA_H_

#include <iostream>
#include "Object.h"
#include "PDBString.h"
#include "PDBVector.h"

//  PRELOAD %CatSyncRequest%

using namespace std;

namespace pdb {

/**
 * This class is used to sync a worker node with the manager
 */
class CatSyncRequest : public Object {
public:

  CatSyncRequest() = default;

  CatSyncRequest(const std::string &nodeID, const std::string &nodeIP, int port, const std::string &nodeType, int32_t numCores, int64_t totalMemory) {

    // init the fields
    this->nodeID = nodeID;
    this->nodeIP = nodeIP;
    this->nodePort = port;
    this->nodeType = nodeType;
    this->numCores = numCores;
    this->totalMemory = totalMemory;
  }

  explicit CatSyncRequest(const Handle<CatSyncRequest> &requestToCopy) {
    nodeID = requestToCopy->nodeID;
    nodeIP = requestToCopy->nodeIP;
    nodePort = requestToCopy->nodePort;
    nodeType = requestToCopy->nodeType;
    numCores = requestToCopy->numCores;
    totalMemory = requestToCopy->totalMemory;
  }

  ~CatSyncRequest() = default;

  ENABLE_DEEP_COPY

  /**
   * ID of the node
   */
  String nodeID;

  /**
   * IP address of the node
   */
  String nodeIP;

  /**
   * The port of the node
   */
  int nodePort = -1;

  /**
   * The type of the node "worker" or "manager"
   */
  String nodeType;

  /**
   * The number of cores on the node
   */
  int32_t numCores = -1;

  /**
   * The amount of memory on the node
   */
  int64_t totalMemory = -1;
};

} /* namespace pdb */

#endif /* CATALOG_NODE_METADATA_H_ */
