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

#ifndef CATALOG_CLU_SYNC_REQUEST_H_
#define CATALOG_CLU_SYNC_REQUEST_H_

#include <iostream>
#include "Object.h"
#include "PDBString.h"
#include "PDBVector.h"

//  PRELOAD %CluSyncRequest%

using namespace std;

namespace pdb {

/**
 * This class is used to sync a worker node with the manager
 */
class CluSyncRequest : public Object {
 public:

  CluSyncRequest() = default;

  CluSyncRequest(const std::string &nodeIP, int port, const std::string &nodeType, int64_t nodeMemory, int32_t nodeNumCores) {

    // init the fields
    this->nodeIP = nodeIP;
    this->nodePort = port;
    this->nodeType = nodeType;
    this->nodeMemory = nodeMemory;
    this->nodeNumCores = nodeNumCores;
  }

  explicit CluSyncRequest(const Handle<CluSyncRequest> &requestToCopy) {
    nodeIP = requestToCopy->nodeIP;
    nodePort = requestToCopy->nodePort;
    nodeType = requestToCopy->nodeType;
    nodeMemory = requestToCopy->nodeMemory;
    nodeNumCores = requestToCopy->nodeNumCores;
  }

  ~CluSyncRequest() = default;

  ENABLE_DEEP_COPY

  /**
   * IP address of the node
   */
  pdb::String nodeIP;

  /**
   * The port of the node
   */
  int nodePort = -1;

  /**
   * The type of the node "worker" or "manager"
   */
  pdb::String nodeType;

  /**
   * The size of the memory on this machine
   */
  int64_t nodeMemory = -1;

  /**
   * The number of cores on this machine
   */
  int32_t nodeNumCores = -1;
};

} /* namespace pdb */

#endif /* CATALOG_NODE_METADATA_H_ */
