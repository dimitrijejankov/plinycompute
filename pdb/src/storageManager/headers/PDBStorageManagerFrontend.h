//
// Created by dimitrije on 2/9/19.
//

#ifndef PDB_STORAGEMANAGERFRONTEND_H
#define PDB_STORAGEMANAGERFRONTEND_H

#include <mutex>

#include <PDBSet.h>
#include <PDBPageCompare.h>
#include <PDBCatalogNode.h>
#include <ServerFunctionality.h>
#include <PDBPageHandle.h>
#include <StoGetPageRequest.h>
#include <StoGetNextPageRequest.h>
#include <StoDispatchData.h>
#include <StoSetStatsRequest.h>

namespace pdb {

class PDBStorageManagerFrontend : public ServerFunctionality {

public:

  virtual ~PDBStorageManagerFrontend();

  /**
   * Initialize the storage manager frontend
   */
  void init() override;

  void registerHandlers(PDBServer &forMe) override;

 private:

  /**
   * This is the response to @see requestPage. Basically it compresses the page and sends it's bytes over the wire to
   * the node that made the request.
   *
   * @tparam Communicator - the communicator class PDBCommunicator is used to handle the request. This is basically here
   * so we could write unit tests
   *
   * @param request - the request for the page we got
   * @param sendUsingMe - the communicator to the node that made the request
   * @return - the result of the handler (success, error)
   */
  template <class Communicator>
  std::pair<bool, std::string> handleGetPageRequest(const pdb::Handle<pdb::StoGetPageRequest> &request, std::shared_ptr<Communicator> &sendUsingMe);

  /**
   * This handler basically accepts the data issued by the dispatcher onto a anonymous page,
   * does some bookkeeping and forwards the page to the backend to be stored
   *
   * @tparam Communicator - the communicator class PDBCommunicator is used to handle the request. This is basically here
   * so we could write unit tests
   *
   * @tparam Requests - the factory class to make request. RequestsFactory class is being used this is just here as a template so we
   * can mock it in the unit tests
   *
   * @param request - the request to handle the dispatched data issued by the @see PDBDistributedStorage of the manager
   * @param sendUsingMe - the communicator to the node that made the request (should be the manager)
   * @return - the result of the handler (success, error)
   */
  template <class Communicator, class Requests>
  std::pair<bool, std::string> handleDispatchedData(pdb::Handle<pdb::StoDispatchData> request, std::shared_ptr<Communicator> sendUsingMe);

  /**
   *
   * @tparam Communicator
   * @tparam Requests
   * @param request
   * @param sendUsingMe
   * @return
   */
  template <class Communicator, class Requests>
  std::pair<bool, std::string> handleGetNumPages(pdb::Handle<pdb::StoSetStatsRequest> request, std::shared_ptr<Communicator> sendUsingMe);

  /**
   * The logger
   */
  PDBLoggerPtr logger;

  /**
   * The last page for a particular set
   */
  map <PDBSetPtr, size_t, PDBSetCompare> lastPages;

  /**
   * Lock last pages
   */
  std::mutex m;
};

}

#include <PDBStorageManagerFrontendTemplate.cc>

#endif //PDB_STORAGEMANAGERFRONTEND_H
