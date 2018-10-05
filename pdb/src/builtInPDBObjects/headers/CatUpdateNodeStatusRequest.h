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

#ifndef CAT_UPDATE_NODE_STATUS_REQ_H
#define CAT_UPDATE_NODE_STATUS_REQ_H

#include "Object.h"
#include "Handle.h"
#include "PDBString.h"

// PRELOAD %CatUpdateNodeStatusRequest%

namespace pdb {

class CatUpdateNodeStatusRequest : public Object {

 public:

  CatUpdateNodeStatusRequest() = default;

  ~CatUpdateNodeStatusRequest() = default;

  CatUpdateNodeStatusRequest(const std::string& nodeID, bool isActive) : nodeID (nodeID), isActive(isActive) {}

  explicit CatUpdateNodeStatusRequest(const Handle<CatUpdateNodeStatusRequest> &requestToCopy) {
    nodeID = requestToCopy->nodeID;
    isActive = requestToCopy->isActive;
  }

  ENABLE_DEEP_COPY

  /**
   * The id of the node we are updating
   */
  String nodeID;

  /**
   * True if the node status is active false otherwise
   */
  bool isActive = false;
};
}

#endif
