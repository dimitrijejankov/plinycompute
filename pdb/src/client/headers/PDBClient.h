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
#ifndef PDBCLIENT_H
#define PDBCLIENT_H

#include "PDBCatalogClient.h"
#include "PDBDistributedStorageClient.h"
#include "ServerFunctionality.h"

#include "Handle.h"
#include "PDBVector.h"
#include "HeapRequest.h"
#include "PDBComputationClient.h"

/**
 * This class provides functionality so users can connect and access
 * different Server functionalities of PDB, such as:
 *
 *  Catalog services
 *  Dispatcher services
 *  Distributed Storage services
 *  Query services
 *
 */

namespace pdb {

class PDBClient : public ServerFunctionality {

 public:

  /**
   * Creates PDBClient
   * @param portIn - the port number of the PDB manager server
   * @param addressIn - the IP address of the PDB manager server
   */
  PDBClient(int portIn, std::string addressIn);

  PDBClient() = default;

  ~PDBClient() = default;

  void registerHandlers(PDBServer &forMe) override {};

  /**
   * Returns the error message generated by a function call
   * @return
   */
  string getErrorMessage();

  /**
   * Sends a request to the Catalog Server to shut down the server that we are connected to
   * returns true on success
   * @return - true if we succeeded false otherwise
   */
  bool shutDownServer();

  /**
   * Creates a database
   * @param databaseName - the name of the database we want to create
   * @return - true if we succeed
   */
  bool createDatabase(const std::string &databaseName);

  /**
   * Creates a set with a given type (using a template) for an existing
   * database, no page_size needed in arg.
   *
   * @tparam DataType - the type of the data the set stores
   * @param databaseName - the name of the database
   * @param setName - the name of the set we want to create
   * @return - true if we succeed
   */
  template<class DataType>
  bool createSet(const std::string &databaseName, const std::string &setName);

  /**
   * Sends a request to the Catalog Server to register a user-defined type defined in a shared library.
   * @param fileContainingSharedLib - the file that contains the library
   * @return
   */
  bool registerType(const std::string &fileContainingSharedLib);

  /**
   * Returns an iterator to the set
   * @tparam DataType - the type of the data of the set
   * @param dbName - the name of the database
   * @param setName - the name of the set
   * @return true if we succeed false otherwise
   */
  template<class DataType>
  PDBStorageIteratorPtr<DataType> getSetIterator(const std::string& dbName, const std::string& setName);

  /**
   * Runs the query with the provided computations and TCAP string, should be used if you want a custom TCAP
   * @param computations - the computations asocciated with the TACP
   * @param tcap - the TCAP string we are running
   * @return true if we succeed in executing
   */
  bool executeComputations(Handle<Vector<Handle<Computation>>> &computations, const pdb::String &tcap);

  bool createIndex(const std::string &db, const std::string &set);

  bool materialize(const std::string &db, const std::string &set, const std::string &pageSet);

  bool broadcast(const std::string &inputPageSet, const std::string& pageSet);

  bool localJoin(const std::string& lhsPageSet, const std::vector<int32_t>& lhs_indices,
                 const std::string& rhsPageSet, const std::vector<int32_t>& rhs_indices,
                 const std::vector<Handle<Computation>> &sinks, const std::string& pageSet,
                 const std::string& startPageSet, const std::string& endPageSet);

  bool shuffle(const std::string &inputPageSet, const std::vector<int32_t>& indices, const std::string& soml);

  bool mm3D(int32_t n, int32_t num_threads, int32_t num_nodes);

  bool localAggregation(const std::string &inputPageSet, const std::vector<int32_t>& indices, const std::string& pageSet);

  /**
   * Runs the query by specifying just the sinks, the tcap will be automatically generated
   * @param computations - the computations
   * @return true if we succeed false otherwise
   */
  bool executeComputations(const std::vector<Handle<Computation>> &sinks);

  /**
   * Lists all metadata registered in the catalog.
   */
  void listAllRegisteredMetadata();

  /**
   * Lists the Databases registered in the catalog.
   */
  void listRegisteredDatabases();

  /**
   * Lists the Sets for a given database registered in the catalog.
   * @param databaseName - the name of the database we want to list the sets for
   */
  void listRegisteredSetsForADatabase(const std::string &databaseName);

  /**
   * Lists the Nodes registered in the catalog.
   */
  void listNodesInCluster();

  /**
   * Lists user-defined types registered in the catalog.
   */
  void listUserDefinedTypes();

  /**
   * Send the data to be stored in a set
   * @param setAndDatabase - the database name and set name
   * @return
   */
  template<class DataType>
  bool sendData(const std::string &database, const std::string &set, Handle<Vector<Handle<DataType>>> dataToSend);

  template<class DataType>
  bool sendData(const std::string &database, const std::string &set, Handle<Vector<Handle<DataType>>> dataToSend, int32_t node);

  bool clearSet(const std::string &dbName, const std::string &setName);

  bool removeSet(const std::string &dbName, const std::string &setName);

 private:

  std::shared_ptr<pdb::PDBCatalogClient> catalogClient;
  std::shared_ptr<pdb::PDBDistributedStorageClient> distributedStorage;
  std::shared_ptr<pdb::PDBComputationClient> computationClient;

  // Port of the PlinyCompute manager node
  int port = -1;

  // IP address of the PlinyCompute manager node
  std::string address;

  // Error Message (if an error occurred)
  std::string errorMsg;

  // Message returned by a PlinyCompute function
  std::string returnedMsg;

  // Client logger
  PDBLoggerPtr logger;
};
}

#include "PDBClientTemplate.cc"
#include "PDBDistributedStorageClientTemplate.cc"

#endif
