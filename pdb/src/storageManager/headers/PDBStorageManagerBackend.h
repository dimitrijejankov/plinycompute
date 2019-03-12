#ifndef PDB_PDBStorageManagerBackend_H
#define PDB_PDBStorageManagerBackend_H

#include <ServerFunctionality.h>
#include <StoStoreOnPageRequest.h>
#include "PDBAbstractPageSet.h"
#include "PDBSetPageSet.h"
#include "PDBAnonymousPageSet.h"

namespace pdb {

class PDBStorageManagerBackend : public ServerFunctionality {

public:

  void registerHandlers(PDBServer &forMe) override;

  void init() override;

  /**
   * This method contacts the frontend to get a PageSet for a particular PDB set
   * @param db - the database the set belongs to
   * @param set - the set name
   * @return the PDBPage set
   */
  PDBSetPageSetPtr createPageSetFromPDBSet(const std::string &db,
                                           const std::string &set,
                                           const std::pair<uint64_t, std::string> &pageSetID);

  /**
   *
   * @param pageSetID
   * @return
   */
  PDBAnonymousPageSetPtr createAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID);

  /**
   *
   * @param pageSetID
   * @return
   */
  PDBAbstractPageSetPtr getPageSet(const std::pair<uint64_t, std::string> &pageSetID);

  /**
   *
   * @param pageSet
   * @param set
   * @return
   */
  bool materializePageSet(PDBAbstractPageSetPtr pageSet, const std::pair<std::string, std::string> &set);

 private:

  /**
   * The logger
   */
  PDBLoggerPtr logger;

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

  /**
   * The page sets that are on the backend
   */
  map<std::pair<uint64_t, std::string>, PDBAbstractPageSetPtr> pageSets;

  /**
   * the mutex to lock the page sets
   */
  std::mutex pageSetMutex;
};

using PDBStorageManagerBackendPtr = std::shared_ptr<PDBStorageManagerBackend>;

}

#endif

#include <PDBStorageManagerBackendTemplate.cc>