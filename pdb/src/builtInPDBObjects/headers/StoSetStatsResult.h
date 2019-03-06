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

#pragma once

#include "Object.h"
#include "Handle.h"
#include "PDBString.h"

// PRELOAD %StoSetStatsResult%

namespace pdb {

// encapsulates a request to add data to a set in storage
class StoSetStatsResult : public Object {

public:

  StoSetStatsResult() = default;
  ~StoSetStatsResult() = default;

  StoSetStatsResult(uint64_t numPages, uint64_t size, bool success) : numPages(numPages), size(size), success(success) {}

  ENABLE_DEEP_COPY

  /**
   * the number of pages on this noode
   */
  uint64_t numPages = 0;

  /**
   * the size of the part of the set stored on this node
   */
  uint64_t size = 0;

  /**
   * was the request a success
   */
  bool success = false;

};

}