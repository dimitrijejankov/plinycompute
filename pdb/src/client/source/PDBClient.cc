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

#ifndef PDBCLIENT_CC
#define PDBCLIENT_CC

#include <ShutDown.h>
#include <PDBClient.h>
#include <QueryGraphAnalyzer.h>
#include <physicalAlgorithms/TRABroadcast.h>
#include <CSExecuteComputation.h>
#include <physicalAlgorithms/TRALocalJoin.h>

#include "PDBClient.h"

namespace pdb {

PDBClient::PDBClient(int portIn, std::string addressIn) : port(portIn), address(addressIn) {

  // init the logger
  logger = make_shared<PDBLogger>("clientLog");

  // init the catalog client
  catalogClient = std::make_shared<pdb::PDBCatalogClient>(portIn, addressIn, logger);

  // init the distributed storage client
  distributedStorage = std::make_shared<pdb::PDBDistributedStorageClient>(portIn, addressIn, logger);

  // init the computation client
  computationClient = std::make_shared<pdb::PDBComputationClient>(addressIn, portIn, logger);
}

string PDBClient::getErrorMessage() {
  return errorMsg;
}

/****
 * Methods for invoking DistributedStorageManager-related operations
 */
bool PDBClient::createDatabase(const std::string &databaseName) {

  bool result = catalogClient->createDatabase(databaseName, returnedMsg);

  if (!result) {

    errorMsg = "Not able to create database: " + returnedMsg;

  } else {
    cout << "Created database.\n";
  }
  return result;
}

// makes a request to shut down a PDB server /// TODO this should be moved
bool PDBClient::shutDownServer() {

  // get the workers
  auto workers = catalogClient->getActiveWorkerNodes();

  // shutdown the workers
  bool success = true;
  for(const auto &w : workers) {
    success = success && RequestFactory::heapRequest<ShutDown, SimpleRequestResult, bool>(logger, w->port, w->address, false, 1024,
     [&](Handle<SimpleRequestResult> result) {

       // do we have a result
       if(result == nullptr) {

         errorMsg = "Error getting type name: got nothing back from catalog";
         return false;
       }

       // did we succeed
       if (!result->getRes().first) {

         errorMsg = "Error shutting down server: " + result->getRes().second;
         logger->error("Error shutting down server: " + result->getRes().second);

         return false;
       }

       // we succeeded
       return true;
     });
  }

  // shutdown
  return success && RequestFactory::heapRequest<ShutDown, SimpleRequestResult, bool>(logger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {

        // do we have a result
        if(result == nullptr) {

          errorMsg = "Error getting type name: got nothing back from catalog";
          return false;
        }

        // did we succeed
        if (!result->getRes().first) {

          errorMsg = "Error shutting down server: " + result->getRes().second;
          logger->error("Error shutting down server: " + result->getRes().second);

          return false;
        }

        // we succeeded
        return true;
      });
}


bool PDBClient::registerType(const std::string &fileContainingSharedLib) {

  bool result = catalogClient->registerType(fileContainingSharedLib, returnedMsg);
  if (!result) {
    errorMsg = "Not able to register type: " + returnedMsg;
    exit(-1);
  } else {
    cout << "Type has been registered.\n";
  }
  return result;
}

bool PDBClient::clearSet(const string &dbName, const string &setName) {
  return distributedStorage->clearSet(dbName, setName, errorMsg);
}

bool PDBClient::removeSet(const string &dbName, const string &setName) {
  return distributedStorage->removeSet(dbName, setName, errorMsg);
}

bool PDBClient::executeComputations(Handle<Vector<Handle<Computation>>> &computations, const pdb::String &tcap) {
  return computationClient->executeComputations(computations, tcap, errorMsg);
}

bool PDBClient::executeComputations(const std::vector<Handle<Computation>> &sinks) {

  // create the graph analyzer
  pdb::QueryGraphAnalyzer queryAnalyzer(sinks);

  // here is the list of computations
  Handle<Vector<Handle<Computation>>> myComputations = makeObject<Vector<Handle<Computation>>>();

  // parse the TCAP string
  std::string TCAPString = queryAnalyzer.parseTCAPString(*myComputations);
  std::cout << TCAPString << "\n";

  // execute the computations
  return computationClient->executeComputations(myComputations, TCAPString, errorMsg);
}

void PDBClient::listAllRegisteredMetadata() {
  cout << catalogClient->listAllRegisteredMetadata(returnedMsg);
}

void PDBClient::listRegisteredDatabases() {
  cout << catalogClient->listRegisteredDatabases(returnedMsg);
}

void PDBClient::listRegisteredSetsForADatabase(const std::string &databaseName) {
  cout << catalogClient->listRegisteredSetsForADatabase(databaseName, returnedMsg);
}

void PDBClient::listNodesInCluster() {
  cout << catalogClient->listNodesInCluster(returnedMsg);
}

void PDBClient::listUserDefinedTypes() {
  cout << catalogClient->listUserDefinedTypes(returnedMsg);
}

bool PDBClient::broadcast(const std::string &db, const std::string &set, const std::string& pageSet) {

  pdb::Handle<TRABroadcast> alg = pdb::makeObject<TRABroadcast>(db, set, pageSet);

  // essentially the buffer should be of this size
  auto bufferSize = 1024u * 1024u;

  // increment the buffer in increments of
  while(bufferSize < 100 * 1024u * 1024u) {

    try {

      // send the request
      std::string error;
      return RequestFactory::heapRequest<CSExecuteComputation, SimpleRequestResult, bool>(logger, port, address, false, bufferSize,
        [&](const Handle<SimpleRequestResult>& result) {

          // check the response
          if ((result != nullptr && !result->getRes().first) || result == nullptr) {

            // log the error
            logger->error("Error executing computations: " + result->getRes().second);
            error = "Error executing computations: " + result->getRes().second;

            // we are done here
            return false;
          }

          // awesome we finished
          return true;
        }, bufferSize, alg);
    }
    catch(pdb::NotEnoughSpace &n) {

      // increment the buffer
      bufferSize += 1024 * 1024;
    }
  }

  // finish since the computation was just too large
  return false;
}

bool PDBClient::localJoin(const std::string &lhsPageSet,
                          const std::vector<int32_t>& lhs_indices,
                          const std::string &rhsDb,
                          const std::string &rhsSet,
                          const std::vector<int32_t>& rhs_indices,
                          const vector<Handle<Computation>> &sinks,
                          const std::string &pageSet) {

  // create the graph analyzer
  pdb::QueryGraphAnalyzer queryAnalyzer(sinks);

  // here is the list of computations
  Handle<Vector<Handle<Computation>>> myComputations = makeObject<Vector<Handle<Computation>>>();

  // parse the TCAP string
  std::string TCAPString = queryAnalyzer.parseTCAPString(*myComputations);
  pdb::Handle<TRALocalJoin> alg = pdb::makeObject<TRALocalJoin>(lhsPageSet, lhs_indices,
                                                                rhsDb, rhsSet, rhs_indices, pageSet);

  // essentially the buffer should be of this size
  auto bufferSize = 1024u * 1024u;

  // increment the buffer in increments of
  while(bufferSize < 100 * 1024u * 1024u) {

    try {

      // send the request
      std::string error;
      return RequestFactory::heapRequest<CSExecuteComputation, SimpleRequestResult, bool>(logger, port, address, false, bufferSize,
        [&](const Handle<SimpleRequestResult>& result) {

          // check the response
          if ((result != nullptr && !result->getRes().first) || result == nullptr) {

            // log the error
            logger->error("Error executing computations: " + result->getRes().second);
            error = "Error executing computations: " + result->getRes().second;

            // we are done here
            return false;
          }

          // awesome we finished
          return true;
        }, myComputations, TCAPString, bufferSize, alg);
    }
    catch(pdb::NotEnoughSpace &n) {

      // increment the buffer
      bufferSize += 1024 * 1024;
    }
  }

  // finish since the computation was just too large
  return false;
}

bool PDBClient::shuffle(const std::string &inputPageSet, const std::vector<int32_t> indices, const std::string &pageSet) {
  return false;
}

bool PDBClient::localAggregation(const std::string &inputPageSet,
                                 const std::vector<int32_t> indices,
                                 const vector<Handle<Computation>> &sinks,
                                 const std::string &pageSet) {
  return false;
}

}

#endif
