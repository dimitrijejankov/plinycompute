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

#ifndef PDB_PDBDISTRIBUTEDSTORAGETEMPLATE_H
#define PDB_PDBDISTRIBUTEDSTORAGETEMPLATE_H

#include "ServerFunctionality.h"
#include "PDBLogger.h"
#include "PDBWork.h"
#include "UseTemporaryAllocationBlock.h"
#include "PDBVector.h"
#include "StoGetNextPageRequest.h"
#include "DisAddData.h"
#include "PDBDispatchPolicy.h"

#include <string>
#include <queue>
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

 private:

  /**
   * The policy we want to use for dispatching.
   * Maybe make this be per set..
   */
  PDBDispatcherPolicyPtr policy;

  /**
   * The logger for the distributed storage
   */
  PDBLoggerPtr logger;
};
}

#include <PDBDistributedStorageTemplate.cc>

#endif  // OBJECTQUERYMODEL_DISPATCHER_H
