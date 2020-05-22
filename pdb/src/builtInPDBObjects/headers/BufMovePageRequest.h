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
#include "PDBSet.h"
#include "BufManagerRequestBase.h"

// PRELOAD %BufMovePageRequest%

namespace pdb {

// encapsulates a request to move an anonymous page to a set
class BufMovePageRequest : public Object {

 public:

  BufMovePageRequest() = default;

  ~BufMovePageRequest() = default;

  BufMovePageRequest(uint64_t pageNumber, uint64_t anonymousPageNumber, const pdb::PDBSetPtr &whichSet) : pageNumber(pageNumber),
                                                                                                          anonymousPageNumber(anonymousPageNumber) {
    dbName = whichSet->getDBName();
    setName = whichSet->getSetName();
  }

  explicit BufMovePageRequest(const pdb::Handle<BufMovePageRequest>& copyMe) {

    // copy stuff
    setName = copyMe->setName;
    dbName = copyMe->dbName;
    pageNumber = copyMe->pageNumber;
    anonymousPageNumber = copyMe->anonymousPageNumber;
  }


  ENABLE_DEEP_COPY

  /**
   * The name of the set we are moving the page to
   */
  String setName;

  /**
   * The name of the database we are moving the page to
   */
  String dbName;

  /**
   * The page number
   */
  uint64_t pageNumber = 0;

  /**
   * Anonymous page number
   */
  uint64_t anonymousPageNumber{};
};
}