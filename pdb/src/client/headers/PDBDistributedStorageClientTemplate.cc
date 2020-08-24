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
#ifndef OBJECTQUERYMODEL_DISPATCHERCLIENTTEMPLATE_CC
#define OBJECTQUERYMODEL_DISPATCHERCLIENTTEMPLATE_CC

#include "PDBDistributedStorageClient.h"
#include "DisAddData.h"
#include "DisClearSet.h"
#include "SimpleRequestResult.h"
#include "PDBStorageVectorIterator.h"
#include "PDBStorageMapIterator.h"


namespace pdb {

template<class DataType>
bool PDBDistributedStorageClient::sendData(const std::string &db,
                                           const std::string &set,
                                           Handle<Vector<Handle<DataType>>> dataToSend,
                                           std::string &errMsg) {


  return RequestFactory::dataHeapRequest<DisAddData, DataType, SimpleRequestResult, bool>(
      *conMgr, port, address, false, 1024, [&](const Handle<SimpleRequestResult>& result) {

        // check the response
        if (result != nullptr && !result->getRes().first) {

          logger->error("Error sending data: " + result->getRes().second);
          errMsg = "Error sending data: " + result->getRes().second;
        }

        return true;
      },
      dataToSend, db, set, getTypeName<DataType>(), false);
}

template<class DataType>
bool PDBDistributedStorageClient::sendDataWithKey(const std::string &db,
                                                  const std::string &set,
                                                  Handle<Vector<Handle<DataType>>> dataToSend,
                                                  std::string &errMsg) {

  return RequestFactory::dataKeyHeapRequest<DisAddData, DataType, SimpleRequestResult, bool>(
      *conMgr, port, address, false, 1024, [&](const Handle<SimpleRequestResult>& result) {

        // check the response
        if (result != nullptr && !result->getRes().first) {

          logger->error("Error sending data: " + result->getRes().second);
          errMsg = "Error sending data: " + result->getRes().second;
        }

        return true;
      },
      dataToSend, db, set, getTypeName<DataType>(), true);
}

template<class DataType>
PDBStorageIteratorPtr<DataType> PDBDistributedStorageClient::getVectorIterator(const std::string &database, const std::string &set) {

  return std::make_shared<PDBStorageVectorIterator<DataType>>(conMgr, address, port, 5, set, database, logger);
}

template<class DataType>
PDBStorageIteratorPtr<DataType> PDBDistributedStorageClient::getMapIterator(const std::string &database, const std::string &set) {

  return std::make_shared<PDBStorageMapIterator<DataType>>(conMgr, address, port, 5, set, database, logger);
}

}

#endif
