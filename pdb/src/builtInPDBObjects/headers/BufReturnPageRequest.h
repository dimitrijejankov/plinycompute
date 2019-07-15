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

#ifndef STO_FREE_PAGE_REQ_H
#define STO_FREE_PAGE_REQ_H

#include "Object.h"
#include "Handle.h"
#include "PDBString.h"
#include "BufManagerRequestBase.h"
#include "../../objectModel/headers/PDBString.h"

// PRELOAD %BufReturnPageRequest%

namespace pdb {

// request to get an anonymous page
class BufReturnPageRequest : public BufManagerRequestBase {

public:

  BufReturnPageRequest(const std::string &setName, const std::string &databaseName, const size_t &pageNumber, bool isDirty)
      : databaseName(databaseName), setName(setName), pageNumber(pageNumber), isDirty(isDirty) {}

  BufReturnPageRequest() = default;

  explicit BufReturnPageRequest(const pdb::Handle<BufReturnPageRequest>& copyMe) : BufManagerRequestBase(*copyMe) {

    // copy stuff
    databaseName = copyMe->databaseName;
    setName = copyMe->setName;
    pageNumber = copyMe->pageNumber;
    isDirty = copyMe->isDirty;
  }

  ~BufReturnPageRequest() = default;

  ENABLE_DEEP_COPY;

  /**
   * The database name
   */
  pdb::String databaseName;

  /**
   * The set name
   */
  pdb::String setName;

  /**
   * The page number
   */
  size_t pageNumber = 0;

  /**
   * Is the page we are returning dirty
   */
  bool isDirty = false;
};
}

#endif
