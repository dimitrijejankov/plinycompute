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

// PRELOAD %StoStartFeedingPageSetRequest%

namespace pdb {

// encapsulates a request to add data to a set in storage
class StoStartFeedingPageSetRequest : public Object {

public:

  StoStartFeedingPageSetRequest() = default;
  ~StoStartFeedingPageSetRequest() = default;

  StoStartFeedingPageSetRequest(const std::pair<uint64_t, std::string> &pageSetID, uint64_t numberOfProcessingThreads, uint64_t numberOfNodes) :
                                computationID(pageSetID.first), tupleSetID(pageSetID.second), numberOfProcessingThreads(numberOfProcessingThreads), numberOfNodes(numberOfNodes) {}

  ENABLE_DEEP_COPY

  /**
   * Returns the page set id
   * @return
   */
  std::pair<uint64_t, std::string> getPageSetID() {
    return std::make_pair(computationID, tupleSetID);
  }

  /**
   * The computation id of the page set
   */
  uint64_t computationID;

  /**
   * The tuple set id of the page set
   */
  String tupleSetID;

  /**
   *
   */
  uint64_t numberOfProcessingThreads;

  /**
   *
   */
  uint64_t numberOfNodes;

};

}