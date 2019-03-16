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

// PRELOAD %StoGetSetPagesResult%

namespace pdb {

// encapsulates a request to add data to a set in storage
class StoGetSetPagesResult : public Object {

public:

  StoGetSetPagesResult() = default;
  ~StoGetSetPagesResult() = default;

  StoGetSetPagesResult(std::vector<uint64_t> &pages, bool success) : pages(pages.size(), 0), success(success) {

    // copy the stuff
    for(auto page : pages) { this->pages.push_back(page);}
  }

  ENABLE_DEEP_COPY

  /**
   * the number of pages on this noode
   */
  pdb::Vector<uint64_t> pages;

  /**
   * was the request a success
   */
  bool success = false;

};

}