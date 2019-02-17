//
// Created by dimitrije on 2/11/19.
//

#ifndef PDB_STORAGEMANAGERBACKEND_H
#define PDB_STORAGEMANAGERBACKEND_H

#include <ServerFunctionality.h>
#include <StoStoreOnPageRequest.h>

namespace pdb {

class PDBStorageManagerBackend : public ServerFunctionality {

public:

  void registerHandlers(PDBServer &forMe) override;

private:

  /**
   * This method simply stores the data that follows the request onto a page.
   * The data is compressed it is uncompressed to the page
   *
   * @tparam Communicator - the communicator class PDBCommunicator is used to handle the request. This is basically here
   * so we could write unit tests
   *
   * @param request - the request we got
   * @param sendUsingMe - the communicator to the node that made the request. In this case this is the communicator to the frontend.
   * @return the result of the handler (success, error)
   */
  template <class Communicator>
  std::pair<bool, std::string> handleStoreOnPage(const pdb::Handle<pdb::StoStoreOnPageRequest> &request, std::shared_ptr<Communicator> &sendUsingMe);

};

}

#include <PDBStorageManagerBackendTemplate.h>

#endif //PDB_STORAGEMANAGERBACKEND_H
