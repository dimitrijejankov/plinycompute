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

// PRELOAD %CatSetIncrementSetRecordInfo%

namespace pdb {

/**
 * Encapsulates a request to update the size of a set
 */
class CatSetIncrementSetRecordInfo : public Object {

public:

  CatSetIncrementSetRecordInfo() = default;
  CatSetIncrementSetRecordInfo(const string &nodeId,
                               const string &databaseName,
                               const string &setName,
                               size_t sizeAdded,
                               size_t recordsStored,
                               size_t keySizeAdded,
                               size_t keyRecordsStored) : nodeID(nodeId),
                                                          databaseName(databaseName),
                                                          setName(setName),
                                                          sizeAdded(sizeAdded),
                                                          recordsStored(recordsStored),
                                                          keySizeAdded(keySizeAdded),
                                                          keyRecordsStored(keyRecordsStored) {}

  ~CatSetIncrementSetRecordInfo() = default;

  /**
   * Copy the request this is needed by the broadcast
   * @param pdbItemToCopy - the request to copy
   */
  explicit CatSetIncrementSetRecordInfo(const Handle<CatSetIncrementSetRecordInfo>& pdbItemToCopy) {

    // copy the thing
    nodeID = pdbItemToCopy->nodeID;
    databaseName = pdbItemToCopy->databaseName;
    setName = pdbItemToCopy->setName;
    sizeAdded = pdbItemToCopy->sizeAdded;
    recordsStored = pdbItemToCopy->recordsStored;
    keySizeAdded = pdbItemToCopy->keySizeAdded;
    keyRecordsStored = pdbItemToCopy->keyRecordsStored;
  }

  ENABLE_DEEP_COPY

  /**
   * The ID of the node where stuff is stored
   */
  String nodeID;

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
  size_t sizeAdded{0};

  /**
   * The number of records stored
   */
  size_t recordsStored{0};

  /**
   * The size of the update in bytes
   */
  size_t keySizeAdded{0};

  /**
   * The number of records stored
   */
  size_t keyRecordsStored{0};
};
}