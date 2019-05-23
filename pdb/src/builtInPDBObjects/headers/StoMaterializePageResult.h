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

// PRELOAD %StoMaterializePageResult%

namespace pdb {

// encapsulates a request to add data to a set in storage
class StoMaterializePageResult : public Object {

public:

  StoMaterializePageResult() = default;
  ~StoMaterializePageResult() = default;

  StoMaterializePageResult(const std::string &db, const std::string &set, size_t materializeSize, bool success, bool hasNext) :
                           materializeSize(materializeSize), databaseName(db), setName(set), success(success), hasNext(hasNext) {}

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
   * The pairs of <page number, size> that we were writing to
   */
  size_t materializeSize = 0;

  /**
   * Did we succeed in writing the stuff to the page
   */
  bool success = false;

  /**
   * Do we have a next page or not
   */
  bool hasNext = false;
};

}