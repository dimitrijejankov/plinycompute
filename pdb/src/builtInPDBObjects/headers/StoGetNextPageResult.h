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


#ifndef OBJECTQUERYMODEL_StoGetNextPageResult_H
#define OBJECTQUERYMODEL_StoGetNextPageResult_H

#include "Object.h"
#include "Handle.h"
#include "PDBString.h"

// PRELOAD %StoGetNextPageResult%

namespace pdb {

// encapsulates a request to add data to a set in storage
class StoGetNextPageResult : public Object {

public:

  StoGetNextPageResult() = default;
  ~StoGetNextPageResult() = default;

  StoGetNextPageResult(const uint64_t page, const std::string &nodeID, uint64_t pageSize, bool hasNext)
      : page(page), nodeID(nodeID), hasNext(hasNext), pageSize(pageSize) {}

  ENABLE_DEEP_COPY

  /**
   * page of the set where we are storing the stuff
   */
  uint64_t page = 0;

  /**
   * The id of the node
   */
  String nodeID;

  /**
   * The size of the page
   */
  uint64_t pageSize = 0;

  /**
   * Do we have another one
   */
  bool hasNext = false;
};

}

#endif
