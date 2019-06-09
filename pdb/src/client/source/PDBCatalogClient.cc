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

#ifndef CATALOG_CLIENT_CC
#define CATALOG_CLIENT_CC

#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <CatGetWorkersRequest.h>
#include <CatUpdateNodeStatusRequest.h>
#include <PDBCatalogClient.h>
#include <CatSetUpdateSizeRequest.h>
#include <CatSetUpdateContainerTypeRequest.h>

#include "CatCreateDatabaseRequest.h"
#include "CatCreateSetRequest.h"
#include "CatDeleteDatabaseRequest.h"
#include "CatDeleteSetRequest.h"
#include "CatRegisterType.h"
#include "CatSetObjectTypeRequest.h"
#include "CatGetType.h"
#include "CatGetSetRequest.h"
#include "CatGetSetResult.h"
#include "CatGetDatabaseRequest.h"
#include "CatGetDatabaseResult.h"
#include "CatPrintCatalogResult.h"
#include "CatTypeNameSearchResult.h"
#include "CatGetTypeResult.h"
#include "CatGetTypeResult.h"
#include "PDBCatalogClient.h"
#include "ShutDown.h"
#include "SimpleSendDataRequest.h"
#include "CatUserTypeMetadata.h"
#include "CatGetWorkersResult.h"

namespace pdb {

// Constructor
PDBCatalogClient::PDBCatalogClient(int portIn, std::string addressIn, PDBLoggerPtr myLoggerIn) {

  // get the communicator information
  port = portIn;
  address = std::move(addressIn);
  myLogger = myLoggerIn;

  // and let the v-table map know this information
  if (!theVTable->getCatalogClient()) {
    theVTable->setCatalogClient(this);
  }

  // set up the mutex
  pthread_mutex_init(&workingMutex, nullptr);
}

// destructor
PDBCatalogClient::~PDBCatalogClient() {
  // Clean up the VTable catalog ptr if it is using this PDBCatalogClient
  if (theVTable->getCatalogClient() == this) {
    theVTable->setCatalogClient(nullptr);
  }

  pthread_mutex_destroy(&workingMutex);
}

/* no handlers for a catalog client!! */
void PDBCatalogClient::registerHandlers(PDBServer &forMe) {}

// sends a request to a Catalog Server to register a Data Type defined in a Shared Library
bool PDBCatalogClient::registerType(std::string fileContainingSharedLib, std::string &errMsg) {

  const LockGuard guard{workingMutex};

  // first, load up the shared library file
  std::ifstream in(fileContainingSharedLib, std::ifstream::ate | std::ifstream::binary);

  // could not open the file
  if (in.fail()) {
    errMsg = "The file " + fileContainingSharedLib + " doesn't exist or cannot be opened.\n";
    return false;
  }

  // grab the file length
  auto fileLen = (size_t) in.tellg();

  // go to the beginning
  in.seekg (0, std::ifstream::beg);

  // load the thing
  auto fileBytes = new char[fileLen];
  in.readsome(fileBytes, fileLen);

  bool res = RequestFactory::heapRequest< CatRegisterType, SimpleRequestResult, bool>(myLogger, port, address, false, 1024 + fileLen,
      [&](Handle<SimpleRequestResult> result) {

        if (result != nullptr) {
          if (!result->getRes().first) {
            errMsg = "Error shutting down server: " + result->getRes().second;
            myLogger->error(errMsg);
            return false;
          }
          return true;
        }

        errMsg = "Error getting type name: got nothing back from catalog";
        return false;

      }, fileBytes, fileLen);

  // free the stuff we loaded
  delete[] fileBytes;

  // return whether we succeeded
  return res;
}


// searches for a User-Defined Type give its name and returns it's TypeID
PDBCatalogTypePtr PDBCatalogClient::getType(const std::string &typeName, std::string &error) {

  PDB_COUT << "Searching for type with the name : " << typeName << "\n";
  return RequestFactory::heapRequest< CatGetType, CatGetTypeResult, PDBCatalogTypePtr>(
      myLogger, port, address, nullptr, 1024 * 1024,
      [&](Handle<CatGetTypeResult> result) {
        if (result != nullptr) {
          PDB_COUT << "Got a type with the type id :" << result->typeID << "\n";
          return std::make_shared<PDBCatalogType>(result->typeID, (std::string) result->typeCategory, result->typeName, std::vector<char>());
        } else {
          PDB_COUT << "searchForObjectTypeName: error in getting typeId\n";
          return (PDBCatalogTypePtr) nullptr;
        }
      },
      typeName);
}

std::vector<pdb::PDBCatalogNodePtr> pdb::PDBCatalogClient::getActiveWorkerNodes() {

  return std::move(RequestFactory::heapRequest< CatGetWorkersRequest, CatGetWorkersResult, std::vector<pdb::PDBCatalogNodePtr>>(
      myLogger, port, address, std::vector<pdb::PDBCatalogNodePtr>(), 1024 * 1024, [&](Handle<CatGetWorkersResult> result) {

        if (result == nullptr) {
          PDB_COUT << "Failed to get the workers!" << "\n";
          return std::vector<pdb::PDBCatalogNodePtr>();
        }

        // copy the result
        std::vector<pdb::PDBCatalogNodePtr> ret;
        for(int i = 0; i < result->nodes->size(); ++i) {

          // grab the node info
          auto &node = (*result->nodes)[i];

          // skip this node if it is not active
          if(!node->active) {
            continue;
          }

          ret.push_back(std::make_shared<pdb::PDBCatalogNode>(node->nodeID, node->nodeAddress, node->nodePort, node->nodeType, node->numCores, node->totalMemory, node->active));
        }

        return std::move(ret);
      }));
}

std::vector<pdb::PDBCatalogNodePtr> pdb::PDBCatalogClient::getWorkerNodes() {

  return std::move(RequestFactory::heapRequest< CatGetWorkersRequest, CatGetWorkersResult, std::vector<pdb::PDBCatalogNodePtr>>(
      myLogger, port, address, std::vector<pdb::PDBCatalogNodePtr>(), 1024 * 1024, [&](Handle<CatGetWorkersResult> result) {

        if (result == nullptr) {
          PDB_COUT << "Failed to get the workers!" << "\n";
          return std::vector<pdb::PDBCatalogNodePtr>();
        }

        // copy the result
        std::vector<pdb::PDBCatalogNodePtr> ret;
        for(int i = 0; i < result->nodes->size(); ++i) {

          // grab the node info
          auto &node = (*result->nodes)[i];

          ret.push_back(std::make_shared<pdb::PDBCatalogNode>(node->nodeID, node->nodeAddress, node->nodePort, node->nodeType, node->numCores, node->totalMemory, node->active));
        }

        return std::move(ret);
      }));
}

// retrieves the content of a Shared Library given it's Type Id
bool PDBCatalogClient::getSharedLibrary(int16_t identifier,
                                     std::string sharedLibraryFileName) {

  PDB_COUT << "PDBCatalogClient: getSharedLibrary for id=" << identifier << std::endl;
  string sharedLibraryBytes;
  string errMsg;

  // using an empty Name b/c it's being searched by typeId
  string typeNameToSearch;
  bool res = getSharedLibraryByTypeName(identifier, typeNameToSearch, sharedLibraryFileName, sharedLibraryBytes, errMsg);

  PDB_COUT << "PDBCatalogClient: deleted putResultHere " << errMsg << std::endl;
  return res;
}

// retrieves a Shared Library given it's typeName
bool PDBCatalogClient::getSharedLibraryByTypeName(
    int16_t identifier, std::string &typeNameToSearch,
    std::string sharedLibraryFileName, string &sharedLibraryBytes,
    std::string &errMsg) {

  PDB_COUT << "inside PDBCatalogClient getSharedLibraryByTypeName for type=" << typeNameToSearch << " and id=" << identifier << std::endl;

  return RequestFactory::heapRequest< CatSharedLibraryByNameRequest, CatUserTypeMetadata, bool>(
      myLogger, port, address, false, 1024 * 1024 * 4,
      [&](Handle<CatUserTypeMetadata> result) {

        PDB_COUT << "In PDBCatalogClient- Handling CatSharedLibraryByNameRequest "
                    "received from "
                    "CatalogServer..."
                 << std::endl;
        if (result == nullptr) {
          PDB_COUT << "FATAL ERROR: can't connect to remote server to fetch "
                       "shared library "
                       "for typeId="
                    << identifier << std::endl;
          exit(-1);
        }

        // gets the typeId returned by the Manager Catalog
        auto returnedTypeId = result->typeID;
        PDB_COUT << "Cat Client - Object Id returned " << returnedTypeId
                 << endl;

        if (returnedTypeId == -1) {
          errMsg = "Error getting shared library: type not found in Manager "
                   "Catalog.\n";
          PDB_COUT << "Error getting shared library: type not found in Manager "
                      "Catalog.\n"
                   << endl;
          return false;
        }

        PDB_COUT << "Cat Client - Finally bytes returned " << result->soBytes->size() << endl;

        if (result->soBytes->size() == 0) {
          errMsg = "Error getting shared library, no data returned.\n";
          PDB_COUT << "Error getting shared library, no data returned.\n"
                   << endl;
          return false;
        }

        // gets metadata and bytes of the registered type
        typeNameToSearch = result->typeName.c_str();
        sharedLibraryBytes = string(result->soBytes->c_ptr(), result->soBytes->size());

        PDB_COUT << "   Metadata in Catalog Client " << result->typeID
                 << " | " << result->typeName << " | "
                 << result->typeCategory << endl;

        PDB_COUT << "copying bytes received in CatClient # bytes "
                 << sharedLibraryBytes.size() << endl;

        // just write the shared library to the file
        int filedesc = open(sharedLibraryFileName.c_str(), O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
        PDB_COUT << "Writing file " << sharedLibraryFileName.c_str() << " size is :" << sharedLibraryBytes.size() << "\n";
        write(filedesc, sharedLibraryBytes.c_str(), sharedLibraryBytes.size());
        close(filedesc);

        PDB_COUT << "objectFile is written by PDBCatalogClient" << std::endl;
        return true;
      },
      identifier, typeNameToSearch);
}

// sends a request to obtain the TypeId of a Type, given a database and set
std::string PDBCatalogClient::getObjectType(std::string databaseName,
                                         std::string setName,
                                         std::string &errMsg) {

  return RequestFactory::heapRequest< CatSetObjectTypeRequest, CatTypeNameSearchResult, std::string>(
      myLogger, port, address, "", 1024,
      [&](Handle<CatTypeNameSearchResult> result) {
        if (result != nullptr) {
          auto success = result->wasSuccessful();
          if (!success.first) {
            errMsg = "Error getting type name: " + success.second;
            myLogger->error("Error getting type name: " + success.second);
          } else
            return result->getTypeName();
        }
        errMsg = "Error getting type name: got nothing back from catalog";
        return std::string("");
      },
      databaseName, setName);
}

// sends a request to the Catalog Server to create Metadata for a new Set
bool PDBCatalogClient::createSet(const std::string &typeName, int16_t typeID, const std::string &databaseName,
                              const std::string &setName, std::string &errMsg) {
  PDB_COUT << "PDBCatalogClient: to create set..." << std::endl;
  return RequestFactory::heapRequest< CatCreateSetRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {
        PDB_COUT << "PDBCatalogClient: received response for creating set"
                 << std::endl;
        if (result != nullptr) {
          if (!result->getRes().first) {
            errMsg = "Error creating set: " + result->getRes().second;
            PDB_COUT << "errMsg" << std::endl;
            myLogger->error("Error creating set: " + result->getRes().second);
            return false;
          }
          PDB_COUT << "PDBCatalogClient: created set" << std::endl;
          return true;
        }
        errMsg = "Error getting type name: got nothing back from catalog";
        PDB_COUT << errMsg << std::endl;
        return false;
      },
      databaseName, setName, typeName, typeID);
}

// sends a request to the Catalog Server to create Metadata for a new Database
bool PDBCatalogClient::createDatabase(std::string databaseName,
                                   std::string &errMsg) {

  return RequestFactory::heapRequest< CatCreateDatabaseRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {
        if (result != nullptr) {
          if (!result->getRes().first) {
            errMsg = "Error creating database: " + result->getRes().second;
            myLogger->error("Error creating database: " +
                            result->getRes().second);
            return false;
          }
          return true;
        }
        errMsg = "Error getting type name: got nothing back from catalog";
        return false;
      },
      databaseName);
}

// sends a request to the Catalog Server to remove Metadata for a Set that is removed
bool PDBCatalogClient::removeSet(const std::string &databaseName, const std::string &setName, std::string &errMsg) {

  return RequestFactory::heapRequest<CatDeleteSetRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {
        if (result != nullptr) {
          if (!result->getRes().first) {
            errMsg = "Error deleting set: " + result->getRes().second;
            myLogger->error("Error deleting set: " + result->getRes().second);
            return false;
          }
          return true;
        }
        errMsg = "Error getting type name: got nothing back from catalog";
        return false;
      },
      databaseName, setName);
}

bool PDBCatalogClient::setExists(const std::string &dbName, const std::string &setName) {

  return RequestFactory::heapRequest< CatGetSetRequest, CatGetSetResult, bool>(
      myLogger, port, address, false, 1024 * 1024,
      [&](Handle<CatGetSetResult> result) {
        return result != nullptr && result->databaseName == dbName && result->setName == setName;
      },
      dbName, setName);
}

bool PDBCatalogClient::databaseExists(const std::string &dbName) {

  // make a request and return the value
  return RequestFactory::heapRequest< CatGetDatabaseRequest, CatGetDatabaseResult, bool>(
      myLogger, port, address, false, 1024 * 1024,
      [&](Handle<CatGetDatabaseResult> result) {
        return result != nullptr && result->database == dbName;
      },
      dbName);
}

pdb::PDBCatalogSetPtr PDBCatalogClient::getSet(const std::string &dbName, const std::string &setName, std::string &errMsg) {

  // make a request and return the value
  return RequestFactory::heapRequest< CatGetSetRequest, CatGetSetResult, pdb::PDBCatalogSetPtr>(
              myLogger, port, address, (pdb::PDBCatalogSetPtr) nullptr, 1024,
              [&](Handle<CatGetSetResult> result) {

                // do we have the thing
                if(result != nullptr && result->databaseName == dbName && result->setName == setName) {
                  return std::make_shared<pdb::PDBCatalogSet>(result->databaseName, result->setName, result->type, result->setSize, result->containerType);
                }

                // return a null pointer otherwise
                return (pdb::PDBCatalogSetPtr) nullptr;
              },
              dbName, setName);
}

bool PDBCatalogClient::incrementSetSize(const std::string &databaseName,
                                        const std::string &setName,
                                        size_t sizeToAdd,
                                        std::string &errMsg) {

  // make a request and return the value
  return RequestFactory::heapRequest< CatSetUpdateSizeRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {

        if (result != nullptr) {
          if (!result->getRes().first) {
            errMsg = "Error updating set: " + result->getRes().second;
            myLogger->error("Error updating set: " + result->getRes().second);
            return false;
          }
          return true;
        }
        errMsg = "Error getting set: got nothing back from catalog";
        return false;
      },
      databaseName, setName, sizeToAdd);
}

bool PDBCatalogClient::updateSetContainerType(const string &databaseName,
                                              const string &setName,
                                              PDBCatalogSetContainerType containerType,
                                              string &errMsg) {
  // make a request and return the value
  return RequestFactory::heapRequest<CatSetUpdateContainerTypeRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {
        if (result != nullptr) {
          if (!result->getRes().first) {
            errMsg = "Error updating set: " + result->getRes().second;
            myLogger->error("Error updating set: " + result->getRes().second);
            return false;
          }
          return true;
        }
        errMsg = "Error getting set: got nothing back from catalog";
        return false;
      },
      databaseName, setName, containerType);
}

pdb::PDBCatalogDatabasePtr PDBCatalogClient::getDatabase(const std::string &dbName, std::string &errMsg) {

  // make a request and return the value
  return RequestFactory::heapRequest< CatGetDatabaseRequest, CatGetDatabaseResult, pdb::PDBCatalogDatabasePtr>(
      myLogger, port, address, (pdb::PDBCatalogDatabasePtr) nullptr, 1024,
      [&](Handle<CatGetDatabaseResult> result) {

        // do we have the thing
        if(result != nullptr && result->database == dbName) {
          return std::make_shared<pdb::PDBCatalogDatabase>(result->database, result->createdOn);
        }

        // return a null pointer otherwise
        return (pdb::PDBCatalogDatabasePtr) nullptr;
      },
      dbName);
}

// sends a request to the Catalog Server to remove Metadata for a Database that
// has been deleted
bool PDBCatalogClient::deleteDatabase(const std::string &databaseName, std::string &errMsg) {

  return RequestFactory::heapRequest< CatDeleteDatabaseRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {
        if (result != nullptr) {
          if (!result->getRes().first) {
            errMsg = "Error deleting database: " + result->getRes().second;
            myLogger->error("Error deleting database: " +
                            result->getRes().second);
            return false;
          }
          return true;
        }
        errMsg = "Error getting type name: got nothing back from catalog";
        return false;
      },
      databaseName);
}

// sends a request to the Catalog Server to add metadata about a Node
bool PDBCatalogClient::syncWithNode(PDBCatalogNodePtr nodeData, std::string &errMsg) {

  return RequestFactory::heapRequest< CatSyncRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {
        if (result != nullptr) {
          if (!result->getRes().first) {
            errMsg = "Error registering node metadata: " + result->getRes().second;
            myLogger->error("Error registering node metadata: " + result->getRes().second);
            return false;
          }
          return true;
        }
        errMsg = "Error registering node metadata in the catalog";
        return false;
      },
      nodeData->nodeID, nodeData->address, nodeData->port, nodeData->nodeType, nodeData->numCores, nodeData->totalMemory);
}

bool PDBCatalogClient::updateNodeStatus(const std::string &nodeID, bool nodeActive, std::string &errMsg) {

  // do the request
  return RequestFactory::heapRequest< CatUpdateNodeStatusRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {
        if (result != nullptr) {

          if (!result->getRes().first) {
            errMsg = "Error updating the node status: " + result->getRes().second;
            myLogger->error(errMsg);
            return false;
          }

          return true;
        }

        errMsg = "Error updating the node status!";
        return false;
      },
      nodeID, nodeActive);
}

// sends a request to the Catalog Server to print all metadata newer than a
// given timestamp
string PDBCatalogClient::printCatalogMetadata(pdb::Handle<pdb::CatPrintCatalogRequest> itemToSearch,
                                           std::string &errMsg) {

  PDB_COUT << "Category to print" << itemToSearch->category.c_str() << "\n";

  return RequestFactory::heapRequest< pdb::CatPrintCatalogRequest, CatPrintCatalogResult, string>(
      myLogger, port, address, "", 1024,
      [&](Handle<CatPrintCatalogResult> result) {
        if (result != nullptr) {
          string res = result->output;
          return res;
        }
        errMsg = "Error printing catalog metadata.";
        return errMsg;
      },
      itemToSearch);
}

// sends a request to the Catalog Server to print all metadata for a given category
string PDBCatalogClient::printCatalogMetadata(std::string &categoryToPrint, std::string &errMsg) {

  pdb::Handle<pdb::CatPrintCatalogRequest> itemToPrint = pdb::makeObject<CatPrintCatalogRequest>("", categoryToPrint);

  return RequestFactory::heapRequest< pdb::CatPrintCatalogRequest, CatPrintCatalogResult, string>(myLogger, port, address, "", 1024,
      [&](Handle<CatPrintCatalogResult> result) {
        if (result != nullptr) {
          string resultToPrint = result->output;
          return resultToPrint;
        }
        errMsg = "Error printing catalog metadata.";
        return errMsg;
      },
      itemToPrint);
}

string PDBCatalogClient::listRegisteredDatabases(std::string &errMsg) {

  string category = "databases";
  return printCatalogMetadata(category, errMsg);
}

string PDBCatalogClient::listRegisteredSetsForADatabase(const std::string &databaseName, std::string &errMsg) {

  string category = "sets";
  return printCatalogMetadata(category, errMsg);
}

string PDBCatalogClient::listNodesInCluster(std::string &errMsg) {

  string category = "nodes";
  return printCatalogMetadata(category, errMsg);
}

string PDBCatalogClient::listUserDefinedTypes(std::string &errMsg) {

  string category = "udts";
  return printCatalogMetadata(category, errMsg);
}

string PDBCatalogClient::listAllRegisteredMetadata(std::string &errMsg) {

  string category = "all";
  return printCatalogMetadata(category, errMsg);
}


}
#endif
