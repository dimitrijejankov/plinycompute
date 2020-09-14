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
#ifndef PDB_CLIENT_TEMPLATE_CC
#define PDB_CLIENT_TEMPLATE_CC

#include "PDBClient.h"

namespace pdb {


  template <class DataType>
  bool PDBClient::createSet(const std::string &databaseName, const std::string &setName) {

    bool result = catalogClient->template createSet<DataType>(databaseName, setName, returnedMsg);

    if (!result) {
        errorMsg = "Not able to create set: " + returnedMsg;
    } else {
        cout << "Created set.\n";
    }

    return result;
  }

  template <class DataType>
  bool PDBClient::sendData(const std::string &database, const std::string &set, Handle<Vector<Handle<DataType>>> dataToSend) {

    // check if that data we are sending is capable of
    bool result;

    // figure out whether we can extract the key
    bool isExtractingKey = false;

    // fist check if it has get key
    constexpr bool hasGetKey = pdb::tupleTests::has_get_key<DataType>::value;

    // if it has get key
    if constexpr (hasGetKey) {

      // figure out if the key is a handle
      using keyType = typename std::remove_reference<decltype(((DataType*) nullptr)->getKey())>::type;
      if constexpr (std::is_base_of<HandleBase, keyType>::value) {

        // send the data with key
        result = distributedStorage->sendDataWithKey<DataType>(database, set, dataToSend, returnedMsg, -1);

        // if we failed
        if(!result) {
          errorMsg = "Not able to send data: " + returnedMsg;
        }

        // finish here
        return result;
      }
    }

    result = distributedStorage->sendData<DataType>(database, set, dataToSend, returnedMsg);

    // if we failed
    if(!result) {
      errorMsg = "Not able to send data: " + returnedMsg;
    }

    return result;
  }

  template<class DataType>
  bool PDBClient::sendData(const std::string &database,
                           const std::string &set,
                           Handle<Vector<Handle<DataType>>> dataToSend,
                           int32_t node) {
    // check if that data we are sending is capable of
    bool result;

    // figure out whether we can extract the key
    bool isExtractingKey = false;

    // fist check if it has get key
    constexpr bool hasGetKey = pdb::tupleTests::has_get_key<DataType>::value;

    // if it has get key
    if constexpr (hasGetKey) {

      // figure out if the key is a handle
      using keyType = typename std::remove_reference<decltype(((DataType*) nullptr)->getKey())>::type;
      if constexpr (std::is_base_of<HandleBase, keyType>::value) {

        // send the data with key
        result = distributedStorage->sendDataWithKey<DataType>(database, set, dataToSend, returnedMsg, node);

        // if we failed
        if(!result) {
          errorMsg = "Not able to send data: " + returnedMsg;
        }

        // finish here
        return result;
      }
    }

    result = distributedStorage->sendData<DataType>(database, set, dataToSend, returnedMsg);

    // if we failed
    if(!result) {
      errorMsg = "Not able to send data: " + returnedMsg;
    }

    return result;
  }

  template<class DataType>
  PDBStorageIteratorPtr<DataType> PDBClient::getSetIterator(const std::string& dbName, const std::string& setName) {

    // get the set from the client
    std::string error;
    auto set = catalogClient->getSet(dbName, setName, error);

    // if we did not find a set return a nullptr for the iterator
    if(set == nullptr) {
      return nullptr;
    }

    // check what kind of set we are dealing with
    if(set->containerType == PDBCatalogSetContainerType::PDB_CATALOG_SET_VECTOR_CONTAINER) {

      // returns the vector iterator
      return distributedStorage->getVectorIterator<DataType>(dbName, setName);
    }
    else if(set->containerType == PDBCatalogSetContainerType::PDB_CATALOG_SET_MAP_CONTAINER) {

      // returns the map iterator
      return distributedStorage->getMapIterator<DataType>(dbName, setName);
    }
    else {

      // ok we can only handle vector and map sets this is a problem
      return distributedStorage->getVectorIterator<DataType>(dbName, setName);
    }
  }

}
#endif
