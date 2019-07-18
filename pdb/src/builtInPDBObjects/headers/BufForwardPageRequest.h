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

#ifndef CAT_STO_GET_FORWARD_REQ_H
#define CAT_STO_GET_FORWARD_REQ_H

#include "Object.h"
#include "Handle.h"
#include "PDBString.h"
#include "BufManagerRequestBase.h"

// PRELOAD %BufForwardPageRequest%

namespace pdb {

// encapsulates a request to forward the page
class BufForwardPageRequest : public BufManagerRequestBase {

public:

  BufForwardPageRequest() = default;

  ~BufForwardPageRequest() = default;

  BufForwardPageRequest(const uint64_t &offset, const uint64_t &pageNum, bool isAnonymous,
                        bool sizeFrozen, const uint64_t &startPos, const int64_t &numBytes,
                        const std::string &setName, const std::string &dbName): offset(offset), pageNum(pageNum),
                                                                                isAnonymous(isAnonymous), sizeFrozen(sizeFrozen),
                                                                                startPos(startPos), numBytes(numBytes), setName(setName),
                                                                                dbName(dbName) {}

  ENABLE_DEEP_COPY

  // a pointer to the raw bytes
  uint64_t offset = 0;

  // the page number
  uint64_t pageNum = 0;

  // is this an anonymous page
  bool isAnonymous = true;

  // is the size frozen
  bool sizeFrozen = false;

  // is this page dirty or not
  bool isDirty = false;

  // the start position in the file
  uint64_t startPos = 0;

  // the size of the page
  int64_t numBytes = 0;

  // the name of the set this page belongs to
  pdb::String setName;

  // the database the set belongs to
  pdb::String dbName;
};
}

#endif
