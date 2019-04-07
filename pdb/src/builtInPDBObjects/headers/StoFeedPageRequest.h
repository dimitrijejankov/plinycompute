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

// PRELOAD %StoFeedPageRequest%

namespace pdb {

// encapsulates a request to add data to a set in storage
class StoFeedPageRequest : public Object {

public:

  StoFeedPageRequest() = default;
  ~StoFeedPageRequest() = default;

  StoFeedPageRequest(uint64_t pageSize, bool hasNextPage) : pageSize(pageSize), hasNextPage(hasNextPage) {}

  ENABLE_DEEP_COPY

  /**
   * The size of the next page if any
   */
  uint64_t pageSize = 0;

  /**
   * Do we have the next page?
   */
  bool hasNextPage = false;

};

}