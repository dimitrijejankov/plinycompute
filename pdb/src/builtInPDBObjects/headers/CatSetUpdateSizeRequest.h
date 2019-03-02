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

// PRELOAD %CatSetUpdateSizeRequest%

namespace pdb {

/**
 * Encapsulates a request to update the size of a set
 */
class CatSetUpdateSizeRequest : public Object {

 public:

  CatSetUpdateSizeRequest() = default;
  ~CatSetUpdateSizeRequest() = default;

  /**
   * Creates a request to get the database
   * @param database - the name of database
   */
  explicit CatSetUpdateSizeRequest(const std::string &database, const std::string &set, size_t sizeUpdate) :
                                   databaseName(database), setName(set), sizeUpdate(sizeUpdate) {}

  /**
   * Copy the request this is needed by the broadcast
   * @param pdbItemToCopy - the request to copy
   */
  explicit CatSetUpdateSizeRequest(const Handle<CatSetUpdateSizeRequest>& pdbItemToCopy) {

    // copy the thing
    databaseName = pdbItemToCopy->databaseName;
    setName = pdbItemToCopy->setName;
    sizeUpdate = pdbItemToCopy->sizeUpdate;
  }

  ENABLE_DEEP_COPY

  /**
   * The name of the database
   */
  String databaseName;

  /**
   * The name of the set
   */
  String setName;

  /**
   * The size of the update in bytes
   */
  size_t sizeUpdate;
};
}