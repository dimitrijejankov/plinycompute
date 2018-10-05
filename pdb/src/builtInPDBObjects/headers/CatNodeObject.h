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

#ifndef CATNODEOBJECT_H
#define CATNODEOBJECT_H

#include "Object.h"
#include "Handle.h"
#include "PDBString.h"

// PRELOAD %CatNodeObject%

namespace pdb {

/**
 * Encapsulates a request to search for a type in the catalog
 */
class CatNodeObject : public Object {

public:

  CatNodeObject() = default;
  ~CatNodeObject() = default;

  CatNodeObject(const std::string &address, int port, const std::string &nodeType, int32_t numCores, int64_t totalMemory, bool active) {
    this->nodeID = address + ":" + std::to_string(port);
    this->nodeAddress = address;
    this->nodePort = port;
    this->nodeType = nodeType;
    this->numCores = numCores;
    this->totalMemory = totalMemory;
    this->active = active;
  }

  // Copy constructor
  explicit CatNodeObject(const Handle<CatNodeObject>& pdbItemToCopy) {

    nodeID = pdbItemToCopy->nodeID;
    nodeAddress = pdbItemToCopy->nodeAddress;
    nodePort = pdbItemToCopy->nodePort;
    nodeType = pdbItemToCopy->nodeType;
    numCores = pdbItemToCopy->numCores;
    totalMemory = pdbItemToCopy->totalMemory;
    active = pdbItemToCopy->active;
  }

  ENABLE_DEEP_COPY

  /**
   * The id of the node is a combination of the ip address and the port concatenated by a column
   */
  pdb::String nodeID;

  /**
   * The ip address of the node
   */
  pdb::String nodeAddress;

  /**
   * The port of the node
   */
  int nodePort = -1;

  /**
   * The node type
   */
  pdb::String nodeType;

  /**
   * The number of cores on the node
   */
  int32_t numCores = -1;

  /**
   * The amount of memory on the node
   */
  int64_t totalMemory = -1;

  /**
   * True if the node is still active
   */
  bool active;
};
}

#endif
