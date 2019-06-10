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

#ifndef OBJECTQUERYMODEL_DISTRIBUTESTORAGECLIENT_H
#define OBJECTQUERYMODEL_DISTRIBUTESTORAGECLIENT_H

#include <PDBSet.h>
#include <PDBStorageIterator.h>
#include "ServerFunctionality.h"
#include "Handle.h"
#include "PDBVector.h"
#include "PDBCatalogClient.h"

namespace pdb {

/**
 * this class serves as a distributed storage client to talk with the DistributedStorage
 * to send Vector<Objects> from clients to the distributed storage server.
 */
class PDBDistributedStorageClient : public ServerFunctionality {

public:

  PDBDistributedStorageClient() = default;

  /**
   * Constructor for the client
   * @param portIn - the port of the manager
   * @param addressIn - the address of the manager
   * @param myLoggerIn - the logger of the client
   */
  PDBDistributedStorageClient(int portIn, std::string addressIn, PDBLoggerPtr myLoggerIn)
                      : port(portIn), address(std::move(addressIn)), logger(std::move(myLoggerIn)) {};

  ~PDBDistributedStorageClient() = default;

  /**
   * Registers the handles needed for the server functionality
   * @param forMe
   */
  void registerHandlers(PDBServer &forMe) override {};

  /**
   * Send the data to the distributed storage
   * @param setAndDatabase - the set and database pair where we want to
   * @return true if we succeed false otherwise
   */
  template<class DataType>
  bool sendData(const std::string &db, const std::string &set, Handle<Vector<Handle<DataType>>> dataToSend, std::string &errMsg);

  /**
   * Removes all the data from a set
   * @param dbName - the name of the database
   * @param setName  - the name of the set
   */
  bool clearSet(const string &dbName, const string &setName, std::string &errMsg);

  /**
   * Removes all the data from the set and then removes the actually set
   * @param dbName - the name of the database the set belongs to
   * @param setName - the name of the set we want to remove
   * @param errMsg - the error message
   * @return true if we succeed false otherwise
   */
  bool removeSet(const string &dbName, const string &setName, std::string &errMsg);

  /**
   * Returns an vector iterator that can fetch records from the storage
   * @param set - the set want to grab the iterator for
   * @return the iterator
   */
  template <class DataType>
  PDBStorageIteratorPtr<DataType> getVectorIterator(const std::string &database, const std::string &set);

  /**
   * Returns an map iterator that can fetch records from the storage
   * @param set - the set want to grab the iterator for
   * @return the iterator
   */
  template <class DataType>
  PDBStorageIteratorPtr<DataType> getMapIterator(const std::string &database, const std::string &set);

private:

  /**
   * The port of the manager
   */
  int port = -1;

  /**
   * The address of the manager
   */
  std::string address;

  /**
   * The logger of the client
   */
  PDBLoggerPtr logger;
};

}

#include "PDBDistributedStorageClientTemplate.cc"

#endif  // OBJECTQUERYMODEL_DISTRIBUTESTORAGECLIENT_H
