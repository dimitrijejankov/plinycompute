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


#ifndef OBJECTQUERYMODEL_DISPDISPATCHDATA_H
#define OBJECTQUERYMODEL_DISPDISPATCHDATA_H

#include "Object.h"
#include "Handle.h"
#include "PDBString.h"

// PRELOAD %DispDispatchData%

namespace pdb {

/**
 * This one looks exactly like the add data but it is sent by the @see PDBDispatcherServer
 */
class DispDispatchData : public Object {

public:

  DispDispatchData() = default;
  ~DispDispatchData() = default;

  DispDispatchData(const std::string &databaseName, const std::string &setName, const std::string &typeName)
      : databaseName(databaseName), setName(setName), typeName(typeName) {
  }

  ENABLE_DEEP_COPY

  /**
   * The name of the database the set belongs to
   */
  String databaseName;

  /**
   * The name of the set we are adding the data to
   */
  String setName;

  /**
   * The name of the type we are adding
   */
  String typeName;
};

}

#endif  // OBJECTQUERYMODEL_DISPATCHERADDDATA_H
