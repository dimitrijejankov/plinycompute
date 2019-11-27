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

#include <utility>

#include "Object.h"
#include "Handle.h"
#include "PDBString.h"

// PRELOAD %StoFetchPageSetPagesRequest%

namespace pdb {

// encapsulates a request to add data to a set in storage
class StoFetchPageSetPagesRequest : public Object {

public:

  StoFetchPageSetPagesRequest() = default;
  ~StoFetchPageSetPagesRequest() = default;

  StoFetchPageSetPagesRequest(std::pair<size_t, pdb::String> pageSetIdentifier, bool forKeys)
                            : pageSetIdentifier(std::move(pageSetIdentifier)), forKeys(forKeys) {}


  ENABLE_DEEP_COPY

  /**
   * The identifier of the page set
   */
  std::pair<size_t, pdb::String> pageSetIdentifier;

  /**
   * Should we get the one for the keys
   */
  bool forKeys = false;
};

}