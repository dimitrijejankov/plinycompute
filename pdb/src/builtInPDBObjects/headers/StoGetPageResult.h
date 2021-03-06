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


#ifndef OBJECTQUERYMODEL_StoGetPageResult_H
#define OBJECTQUERYMODEL_StoGetPageResult_H

#include "Object.h"
#include "Handle.h"
#include "PDBString.h"

// PRELOAD %StoGetPageResult%

namespace pdb {

// encapsulates a request to add data to a set in storage
class StoGetPageResult : public Object {

public:

  StoGetPageResult() = default;
  ~StoGetPageResult() = default;

  StoGetPageResult(size_t size, uint64_t pageNumber, bool hasPage) : size(size), hasPage(hasPage), pageNumber(pageNumber) {}

  ENABLE_DEEP_COPY

  /**
   * The number of the page
   */
  uint64_t pageNumber = 0;

  /**
   * page of the set where we are storing the stuff
   */
  uint64_t size = 0;

  /**
   * Do we have a page
   */
  bool hasPage = false;

};

}

#endif
