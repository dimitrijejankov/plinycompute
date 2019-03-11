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

// PRELOAD %StoStartWritingToSetResult%

namespace pdb {

// encapsulates a request to add data to a set in storage
class StoStartWritingToSetResult : public Object {

public:

  StoStartWritingToSetResult() = default;
  ~StoStartWritingToSetResult() = default;

  StoStartWritingToSetResult(uint64_t startPage, bool success) : startPage(startPage), success(success) {}

  ENABLE_DEEP_COPY

  /**
   * The start page
   */
  uint64_t startPage = 0;

  /**
   * was the request a success?
   */
  bool success = false;

};

}