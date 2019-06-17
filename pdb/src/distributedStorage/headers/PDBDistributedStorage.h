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

#include "ServerFunctionality.h"
#include "PDBLogger.h"
#include "PDBWork.h"
#include "UseTemporaryAllocationBlock.h"
#include "PDBVector.h"
#include "StoGetNextPageRequest.h"
#include "DisAddData.h"
#include "DisClearSet.h"
#include "DisRemoveSet.h"
#include "PDBDistributedStorageSetLock.h"
#include "PDBDispatchPolicy.h"
#include <StoDispatchData.h>

#include <string>
#include <queue>
#include <condition_variable>
#include <unordered_map>
#include <vector>
#include <PDBPageHandle.h>

namespace pdb {

// just make a ptr to the distributed storage
class PDBDistributedStorage;
using PDBDistributedStoragePtr = std::shared_ptr<PDBDistributedStorage>;

/**
 * The DispatcherServer partitions and then forwards a Vector of pdb::Objects received from a
 * PDBDispatcherClient to the proper storage servers
 */
class PDBDistributedStorage : public ServerFunctionality {

public:

  ~PDBDistributedStorage() = default;

  /**
   * Initialize the dispatcher
   */
  void init() override;

  /**
   * Inherited function from ServerFunctionality
   * @param forMe
   */
  void registerHandlers(PDBServer &forMe) override;

  /**
   * Blocks until we are granted to use the set. The requestedState is NONE an exception is thrown.
   * The request expires once the lock goes out of scope
   *
   * @param dbName - the name of the database the set belongs to
   * @param setName - the name of the set
   * @param stateRequested - indicates what we are trying to use the set for
   * @return - a lock for the request if it succeeds
   */
  PDBDistributedStorageSetLockPtr useSet(const std::string &dbName,
                                         const std::string &setName,
                                         PDBDistributedStorageSetState stateRequested);

  /**
   * Tries to use the set, for a specified purpose.
   *
   * If the requestedState is WRITING_DATA then the request will be granted if and only if the set is in WRITING_DATA or NONE state.
   * If the requestedState is READING_DATA then the request will be granted if and only if the set is in READING_DATA or NONE state.
   * If the requestedState is CLEARING_DATA then the request will be granted if and only if the set is in NONE state.
   * If the requestedState is NONE an exception is thrown.
   *
   * The request expires once the lock goes out of scope
   *
   * @param dbName - the name of the database the set belongs to
   * @param setName - the name of the set
   * @param stateRequested - indicates what we are trying to use the set for
   * @return - a lock for the request if it succeeds
   */
  PDBDistributedStorageSetLockPtr tryUsingSet(const std::string &dbName,
                                              const std::string &setName,
                                              PDBDistributedStorageSetState stateRequested,
                                              std::unique_lock<std::mutex> &lck);

private:

  /**
   * Requests a page from a node and stores it's compressed bytes onto an anonymous page.
   *
   * @tparam Communicator - the communicator class PDBCommunicator is used to handle the request. This is basically here
   * so we could write unit tests
   *
   * @tparam Requests - the factory class to make request. RequestsFactory class is being used this is just here as a template so we
   * can mock it in the unit tests
   *
   * @param node - the node we want to request a page from.
   * @param databaseName - the database the page belongs to
   * @param setName - the set the page belongs to
   * @param page - this is the page we are requesting, if the page is not available but another is, this will be set to
   * the number of that page
   *
   * @return - the page handle of the anonymous page
   */
  template<class Communicator, class Requests>
  std::pair<PDBPageHandle, size_t> requestPage(const PDBCatalogNodePtr &node,
                                               const std::string &databaseName,
                                               const std::string &setName,
                                               uint64_t &page);

  /**
  * This handler is used by the iterator to grab it's next page. It will try to find the next page that is just an
  * increment from the last page on a certain node. If it can not find that page on that node it will go to the next node
  * to see if it has any pages. If it has them it stores it's bytes onto an anonymous page and forwards that to the iterator.
  *
  * @tparam Communicator - the communicator class PDBCommunicator is used to handle the request. This is basically here
  * so we could write unit tests
  *
  * @param request - the request for the page we got
  * @param sendUsingMe - the communicator to the node that made the request
  * @return - the result of the handler (success, error)
  */
  template<class Communicator, class Requests>
  std::pair<bool, std::string> handleGetNextPage(const pdb::Handle<pdb::StoGetNextPageRequest> &request,
                                                 std::shared_ptr<Communicator> &sendUsingMe);

  /**
   * This handler adds data to the distributed storage. Basically it checks whether the size of the sent data can fit
   * on a single page. If it can it finds a node the data should be stored on and forwards the data to it.
   *
   * @tparam Communicator - the communicator class PDBCommunicator is used to handle the request. This is basically here
   * so we could write unit tests
   *
   * @tparam Requests - is the request factory for this
   *
   * @param request - the request that contains the data
   * @param sendUsingMe - the communicator that is sending the data
   * @return - the result of the handler (success, error)
   */
  template<class Communicator, class Requests>
  std::pair<bool, std::string> handleAddData(const pdb::Handle<pdb::DisAddData> &request,
                                             std::shared_ptr<Communicator> &sendUsingMe);

  /**
   * This handler clears a particular set, just the data.
   * It will succeed if the set is not in use.
   *
   * @tparam Communicator - the communicator class PDBCommunicator is used to handle the request. This is basically here
   * so we could write unit tests
   *
   * @tparam Requests - is the request factory for this
   *
   * @param request - the request for the clear set
   * @param sendUsingMe - the communicator that is sending the data
   * @return
   */
  template<class Communicator, class Requests>
  std::pair<bool, std::string> handleClearSet(const pdb::Handle<pdb::DisClearSet> &request,
                                              std::shared_ptr<Communicator> &sendUsingMe);
  /**
   * This handler removes a particular set, it removes all info about it.
   * It will succeed if the set is not in use.
   *
   * @tparam Communicator - the communicator class PDBCommunicator is used to handle the request. This is basically here
   * so we could write unit tests
   *
   * @tparam Requests - is the request factory for this
   *
   * @param request - the request for the remove set
   * @param sendUsingMe - the communicator that is sending the data
   * @return
   */
  template<class Communicator, class Requests>
  std::pair<bool, std::string> handleRemoveSet(const pdb::Handle<pdb::DisRemoveSet> &request,
                                               std::shared_ptr<Communicator> &sendUsingMe);

  /**
   * This structure holds info about how we are currently using the set
   */
  struct PDBDistributedStorageSetStateStruct {

    /**
     * The current state
     */
    PDBDistributedStorageSetState state;

    /**
     * The number of current readers
     */
    int32_t numReaders;

    /**
     * Number of current writers
     */
    int32_t numWriters;

  };

  /**
   * Same as the other @tryUsingSet just without lock parameter
   * @param dbName - the name of the database the set belongs to
   * @param setName - the name of the set
   * @param stateRequested - indicates what we are trying to use the set for
   * @return - a lock for the request if it succeeds
   */
  PDBDistributedStorageSetLockPtr tryUsingSet(const std::string &dbName,
                                              const std::string &setName,
                                              PDBDistributedStorageSetState stateRequested);

  /**
   * This marks that we are done using a particular set
   * @param dbName - the name of the database the set belongs to
   * @param setName - the name of the set
   */
  void finishUsingSet(const std::string &dbName, const std::string &setName, PDBDistributedStorageSetState stateRequested);

  /**
   * This method responds back to the client with an error. This should only be used before the actual data is received
   *
   * @tparam Communicator - the communicator class PDBCommunicator is used to handle the request. This is basically here
   * so we could write unit tests
   *
   * @param sendUsingMe - the communicator that is sending the data
   * @param errMsg - the error message that we want to send back to the client
   */
  template<class Communicator>
  void respondAddDataWithError(shared_ptr<Communicator> &sendUsingMe, std::string &errMsg);

  /**
   * The policy we want to use for dispatching.
   * Maybe make this be per set..
   */
  PDBDispatcherPolicyPtr policy;

  /**
   * The logger for the distributed storage
   */
  PDBLoggerPtr logger;

  /**
   * Contains the info about what sets are currently in use by the system, this is runtime info.
   */
  std::map<std::pair<std::string, std::string>, PDBDistributedStorageSetStateStruct> setStates;

  /**
   * Used to lock the set @see setsInUse
   */
  std::mutex setInUseLck;

  /**
   * Used to block in the case we want to wait for a set to be free
   */
  std::condition_variable cv;

  // add the distributed storage lock as a friend
  friend class PDBDistributedStorageSetLock;
};

}

#include <PDBDistributedStorageTemplate.cc>
