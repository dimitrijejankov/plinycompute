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

#include <PDBCatalogSet.h>
#include "CatSharedLibraryByNameRequest.h"
#include "CatSyncRequest.h"
#include "CatPrintCatalogRequest.h"
#include "PDBLogger.h"
#include "PDBServer.h"
#include "ServerFunctionality.h"
#include "PDBCatalogSetStats.h"

namespace pdb {

class PDBCatalogClient;
using PDBCatalogClientPtr = std::shared_ptr<PDBCatalogClient>;

class PDBCatalogClient : public ServerFunctionality {

public:

  // Destructor
  ~PDBCatalogClient();

  // Default Constructor
  PDBCatalogClient() = default;

  // Creates a Catalog Client given the port and address of the catalog
  // for a local catalog the address is typically "localhost"
  // for a remote catalog, the address is the IP address of the machine where the catalog resides
  PDBCatalogClient(int port, std::string address, PDBLoggerPtr myLogger);

  // Registers event handlers associated with this server functionality */
  void registerHandlers(PDBServer &forMe) override;

  // Uses the name of the object type to find its corresponding typeId, returns -1 if not found
  PDBCatalogTypePtr getType(const std::string &typeName, std::string &error);

  std::vector<pdb::PDBCatalogNodePtr> getActiveWorkerNodes();

  std::vector<pdb::PDBCatalogNodePtr> getWorkerNodes();

  // Retrieves the content of a Shared Library given it's Type Id
  bool getSharedLibrary(int16_t identifier, std::string sharedLibraryFileName);

  // Retrieves the content of a Shared Library along with its registered
  // metadata, given it's typeName. Typically this method is invoked by a remote machine that has no
  // knowledge of the typeID
  bool getSharedLibraryByTypeName(int16_t identifier, std::string &typeName,
                                  std::string sharedLibraryFileName,
                                  string &sharedLibraryBytes,
                                  std::string &errMsg);

  // Sends a request to the Catalog Server to register a type with the catalog
  // returns true on success, false on fail
  bool registerType(const std::string& fileContainingSharedLib, std::string &errMsg);

  // Sends a request to the Catalog Server to return the typeName
  // of a user-defined type in the specified database and set; returns "" on err
  std::string getObjectType(const std::string& databaseName, const std::string& setName,
                            std::string &errMsg);

  // Sends a request to the Catalog Server to Create a new database returns true on success, false on fail
  bool createDatabase(const std::string& databaseName, std::string &errMsg);

  // Sends a request to the Catalog Server to register metadata about a node in the cluster
  // returns the identifier of the node
  int32_t syncWithNode(const PDBCatalogNodePtr& nodeData, std::string &errMsg);

  // Updates the node status either set it to active or not
  bool updateNodeStatus(int32_t nodeID, bool nodeActive, std::string &errMsg);

  // Sends a request to the Catalog Server to Creates a new set for a given
  template <class DataType>
  bool createSet(std::string databaseName, std::string setName, std::string &errMsg);

  // same as above, but here we use the type code
  bool createSet(const std::string &typeName, int16_t typeID, const std::string &databaseName,
                 bool isStoringKeys, const std::string &setName, std::string &errMsg);

  // sends a request to the Catalog Server to delete a set returns true on success, false on fail
  bool removeSet(const std::string &databaseName, const std::string &setName, std::string &errMsg);

  // clears the set
  bool clearSet(const std::string &databaseName, const std::string &setName, std::string &errMsg);

  // Increments the size of a set for a particular set by size
  bool incrementSetRecordInfo(int32_t nodeIdentifier,
                              const std::string &databaseName,
                              const std::string &setName,
                              size_t sizeAdded,
                              size_t recordsStored,
                              std::string &errMsg);

  bool incrementKeyRecordInfo(int32_t nodeIdentifier,
                              const std::string &databaseName,
                              const std::string &setName,
                              size_t sizeAdded,
                              size_t keysStored,
                              std::string &errMsg);

  // Update the container type of a set for a particular set by size
  bool updateSetContainerType(const std::string &databaseName,
                              const std::string &setName,
                              PDBCatalogSetContainerType containerType,
                              std::string &errMsg);

  // Gets the stats of a set in the database
  pdb::PDBCatalogSetStatsPtr getSetStats(const std::string &dbName, const std::string &setName, std::string &errMsg);

  // Sends a request to the Catalog Server to delete a database; returns true on success, false on fail
  bool deleteDatabase(const std::string &databaseName, std::string &errMsg);

  bool setExists(const std::string &dbName, const std::string &setName);

  bool databaseExists(const std::string &dbName);

  pdb::PDBCatalogSetPtr getSet(const std::string &dbName, const std::string &setName, std::string &errMsg);

  pdb::PDBCatalogDatabasePtr getDatabase(const std::string &dbName, std::string &errMsg);

  // Sends a request to the Catalog Server to print the content of the metadata stored in the catalog
  string printCatalogMetadata(pdb::Handle<pdb::CatPrintCatalogRequest> itemToSearch,
                              std::string &errMsg);

  // Sends a request to the Catalog Server to print a category of metadata stored in the catalog
  string printCatalogMetadata(std::string &categoryToPrint,
                              std::string &errMsg);

  // Lists all metadata registered in the catalog.
  string listAllRegisteredMetadata(std::string &errMsg);

  // Lists the Databases registered in the catalog.
  string listRegisteredDatabases(std::string &errMsg);

  // Lists the Sets for a given database registered in the catalog.
  string listRegisteredSetsForADatabase(const std::string &databaseName, std::string &errMsg);

  // Lists the Nodes registered in the catalog.
  string listNodesInCluster(std::string &errMsg);

  // Lists the user-defined types registered in the catalog.
  string listUserDefinedTypes(std::string &errMsg);

private:

  // The IP address where this Catalog Client is connected to
  std::string address;

  // The port where this Catalog Client is connected to
  int port = -1;

  // Logger to debug information
  PDBLoggerPtr myLogger;

  // To ensure serialized access
  pthread_mutex_t workingMutex{};
};
}

#include "PDBCatalogClientTemplate.cc"