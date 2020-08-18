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
 * CatSyncResult.h
 *
 */

#pragma once

#include <iostream>
#include <utility>
#include "Object.h"
#include "PDBString.h"
#include "PDBVector.h"

//  PRELOAD %CatSyncResult%

using namespace std;

namespace pdb {

/**
 * This class is used to sync a worker node with the manager
 */
class CatSyncResult : public Object {
public:

  CatSyncResult() = default;

  CatSyncResult(int32_t nodeID, bool success, std::string error) {

    // init the fields
    this->nodeID = nodeID;
    this->success = success;
    this->error = std::move(error);
  }

  explicit CatSyncResult(const Handle<CatSyncResult> &requestToCopy) {
    this->nodeID = requestToCopy->nodeID;
    this->success = requestToCopy->success;
    this->error = requestToCopy->error;
  }

  ~CatSyncResult() = default;

  ENABLE_DEEP_COPY

  // ID of the node
  int32_t nodeID{};

  // did we succeed?
  bool success{};

  // the error if any
  std::string error;
};

} /* namespace pdb */