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


#ifndef OBJECTQUERYMODEL_StoGetNextPaest_H
#define OBJECTQUERYMODEL_StoGetNextPaest_H

#include "Object.h"
#include "Handle.h"
#include "PDBString.h"

// PRELOAD %StoGetNextPageRequest%

namespace pdb {

// encapsulates a request to add data to a set in storage
class StoGetNextPageRequest : public Object {

public:

  StoGetNextPageRequest() = default;
  ~StoGetNextPageRequest() = default;

  StoGetNextPageRequest(const std::string &databaseName, const std::string &setName, const uint64_t page, const std::string &nodeID, bool isFirst)
      : databaseName(databaseName), setName(setName), page(page), nodeID(nodeID), isFirst(isFirst) {}

  ENABLE_DEEP_COPY

  /**
   * The name of the database the set belongs to
   */
  String databaseName;

  /**
   * The name of the set we are storing the stuff
   */
  String setName;

  /**
   * page of the set where we are storing the stuff
   */
  uint64_t page = 0;

  /**
   * The id of the node
   */
  String nodeID;

  /**
   * Are we requesting a page for the first time
   */
   bool isFirst = false;

};

}

#endif
