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

#include <PDBCatalogSet.h>
#include "Object.h"
#include "Handle.h"
#include "PDBString.h"

// PRELOAD %CatSetUpdateContainerTypeRequest%

namespace pdb {

/**
 * Encapsulates a request to update the type of the container
 */
class CatSetUpdateContainerTypeRequest : public Object {

 public:

  CatSetUpdateContainerTypeRequest() = default;
  ~CatSetUpdateContainerTypeRequest() = default;

  /**
   * Creates a request to get the database
   * @param database - the name of database
   */
  explicit CatSetUpdateContainerTypeRequest(const std::string &database,
                                            const std::string &set,
                                            PDBCatalogSetContainerType containerType) : databaseName(database),
                                                                                     setName(set),
                                                                                     containerType(containerType) {}

  /**
   * Copy the request this is needed by the broadcast
   * @param pdbItemToCopy - the request to copy
   */
  explicit CatSetUpdateContainerTypeRequest(const Handle<CatSetUpdateContainerTypeRequest>& pdbItemToCopy) {

    // copy the thing
    databaseName = pdbItemToCopy->databaseName;
    setName = pdbItemToCopy->setName;
    containerType = pdbItemToCopy->containerType;
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
   * The type of the container
   */
  PDBCatalogSetContainerType containerType = PDBCatalogSetContainerType::PDB_CATALOG_SET_NO_CONTAINER;
};
}