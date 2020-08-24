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

#ifndef CAT_CLIENT_TEMPL_CC
#define CAT_CLIENT_TEMPL_CC

#include "PDBCatalogClient.h"
#include "CatCreateSetRequest.h"
#include "HeapRequest.h"
#include "SimpleRequestResult.h"
#include "JoinTupleTests.h"

namespace pdb {

template<class DataType>
bool PDBCatalogClient::createSet(std::string databaseName, std::string setName, std::string &errMsg) {

  // figure out the type name
  std::string typeName = VTableMap::getInternalTypeName(getTypeName<DataType>());

  // figure out whether we can extract the key
  bool isExtractingKey = false;

  // fist check if it has get key
  constexpr bool hasGetKey = pdb::tupleTests::has_get_key<DataType>::value;

  // if it has get key
  if constexpr (hasGetKey) {

    // get the type of the key and figure out if it is a handle
    using keyType = typename std::remove_reference<decltype(((DataType*) nullptr)->getKey())>::type;
    isExtractingKey = std::is_base_of<HandleBase, keyType>::value;
  }

  // get the type id
  int16_t typeID = VTableMap::getIDByName(VTableMap::getInternalTypeName(getTypeName<DataType>()), false);
  if (typeID == -1) {
    errMsg = "Could not find type " + getTypeName<DataType>();
    return -1;
  }

  // make the request
  return RequestFactory::heapRequest<CatCreateSetRequest, SimpleRequestResult, bool>(
      *conMgr, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {
        if (result != nullptr) {
          if (!result->getRes().first) {
            errMsg = "Error creating set: " + result->getRes().second;
            myLogger->error("Error creating set: " + result->getRes().second);
            return false;
          }
          return true;
        }
        errMsg = "Error getting type name: got nothing back from catalog";
        return false;
      },
      databaseName, setName, typeName, isExtractingKey, typeID);
}
}

#endif
