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

#ifndef CAT_DELETE_SET_H
#define CAT_DELETE_SET_H

#include "Object.h"
#include "PDBString.h"
#include "Handle.h"

// PRELOAD %CatDeleteSetRequest%

namespace pdb {

// encapsulates a request to delete a set
class CatDeleteSetRequest : public Object {
 public:

  ~CatDeleteSetRequest() = default;
  CatDeleteSetRequest() = default;
  CatDeleteSetRequest(const std::string &dbName, const std::string &setName, bool onlyClear)
      : dbName(dbName), setName(setName), onlyClear(onlyClear) {}

  explicit CatDeleteSetRequest(const Handle<CatDeleteSetRequest> &requestToCopy) {
    setName = requestToCopy->setName;
    dbName = requestToCopy->dbName;
    onlyClear = requestToCopy->onlyClear;
  }

  std::pair<std::string, std::string> whichSet() {
    return std::make_pair<std::string, std::string>(dbName, setName);
  }

  ENABLE_DEEP_COPY

  // the name of the database the set belongs to
  String dbName;

  // the name of the set
  String setName;

  // tells the catalog to only clear the set and not completely remove it
  bool onlyClear = false;
};
}

#endif
