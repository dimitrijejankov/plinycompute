#ifndef PDB_PDBStorageManagerBackend_H
#define PDB_PDBStorageManagerBackend_H

#include <ServerFunctionality.h>
#include <StoStoreOnPageRequest.h>
#include "PDBAbstractPageSet.h"
#include "PDBSetPageSet.h"
#include "PDBAnonymousPageSet.h"
#include "StoRemovePageSetRequest.h"
#include "StoStartFeedingPageSetRequest.h"
#include "PDBFeedingPageSet.h"

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
  PDBSetPageSetPtr createPageSetFromPDBSet(const std::string &db, const std::string &set);

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
  PDBFeedingPageSetPtr createFeedingAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID, uint64_t numReaders, uint64_t numFeeders);

  /**
   * Returns a pages set that already exists
   * @param pageSetID - the id of the page set. The usual is (computationID, tupleSetID)
   * @return the tuples set if it exists, null otherwise
   */
  PDBAbstractPageSetPtr getPageSet(const std::pair<uint64_t, std::string> &pageSetID);

  /**
   * Removes the page set from the storage.
   * @param pageSetID
   * @return
   */
  bool removePageSet(const std::pair<uint64_t, std::string> &pageSetID);

  /**
   * This method materializes a particular page set to a particular set. It contacts the frontend and grabs a bunch of pages
   * it assumes that the set we are materializing to exists.
   * @param pageSet - the page set we want to materialize
   * @param set - the set we want to materialize to
   * @return true if it succeeds false otherwise
   */
  bool materializePageSet(const PDBAbstractPageSetPtr& pageSet, const std::pair<std::string, std::string> &set);

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
   *
   * @tparam Communicator
   * @param request
   * @param sendUsingMe
   * @return
   */
  template <class Communicator>
  std::pair<bool, std::string> handlePageSet(const pdb::Handle<pdb::StoRemovePageSetRequest> &request, std::shared_ptr<Communicator> &sendUsingMe);

  /**
   *
   * @tparam Communicator
   * @param request
   * @param sendUsingMe
   * @return
   */
  template <class Communicator>
  std::pair<bool, std::string> handleStartFeedingPageSetRequest(pdb::Handle<pdb::StoStartFeedingPageSetRequest> &request, std::shared_ptr<Communicator> &sendUsingMe);

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