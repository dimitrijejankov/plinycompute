#pragma once

#include "TRAIndex.h"
#include <ServerFunctionality.h>
#include <StoStoreDataRequest.h>
#include <PDBKeyExtractor.h>
#include <physicalAlgorithms/PDBSourcePageSetSpec.h>
#include "PDBAbstractPageSet.h"
#include "PDBSetPageSet.h"
#include "PDBAnonymousPageSet.h"
#include "StoRemovePageSetRequest.h"
#include "StoStartFeedingPageSetRequest.h"
#include "StoStoreKeysRequest.h"
#include "PDBFeedingPageSet.h"
#include "PDBRandomAccessPageSet.h"


namespace pdb {

class PDBStorageManagerBackend : public ServerFunctionality {

public:

  void registerHandlers(PDBServer &forMe) override;

  void init() override;

  // creates an index for a page set
  TRAIndexNodePtr createIndex(const std::pair<uint64_t, std::string> &pageSetID);

  // returns the index
  TRAIndexNodePtr getIndex(const std::pair<uint64_t, std::string> &pageSetID);

  /**
   * This method contacts the frontend to get a PageSet for a particular PDB set
   * @param db - the database the set belongs to
   * @param set - the set name
   * @param isKeyed - are we getting the key version of the set
   * @return the PDBPage set
   */
  PDBSetPageSetPtr createPageSetFromPDBSet(const std::string &db, const std::string &set, bool isKeyed);

  /**
   *
   * @param pageSetID
   * @return
   */
  PDBAnonymousPageSetPtr createAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID);

  /**
   * Create a random access page set.
   * @param pageSetID - the id of the page set
   * @return return the random access page set
   */
  PDBRandomAccessPageSetPtr createRandomAccessPageSet(const std::pair<uint64_t, std::string> &pageSetID);

  /**
   *
   * @param pageSetID
   * @return
   */
  PDBFeedingPageSetPtr createFeedingAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID, uint64_t numReaders, uint64_t numFeeders);

  /**
   * This makes a page set that fetches pages of a given set from a particular node
   * @param database - the name of the database the set belongs to
   * @param set - the name of the set
   * @param isKey - are we requesting the key only version of the set
   * @param ip - the ip of the node we are requesting it from
   * @param port - the port of the node
   * @return the page set
   */
  PDBAbstractPageSetPtr fetchPDBSet(const std::string &database,
                                    const std::string &set,
                                    bool isKey,
                                    const std::string &ip,
                                    int32_t port);

  /**
   * This makes a page se that fetches pages of a given page set from a particular node
   * @param pageSetSpec - the specification of the page set we want to grab pages from
   * @param isKey - are we requesting the key only version of the set
   * @param ip - the ip of the node we are requesting it from
   * @param port - the port of the node
   * @return the page set
   */
  PDBAbstractPageSetPtr fetchPageSet(const PDBSourcePageSetSpec &pageSetSpec,
                                     bool isKey,
                                     const std::string &ip,
                                     int32_t port);

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

  /**
   * This materializes the keys of a particular page set
   * @param pageSet - the page set we want to materialize
   * @param set - the set we want to materialize it to
   * @return true if it succeeds
   */
  bool materializeKeys(const PDBAbstractPageSetPtr& pageSet,
                       const std::pair<std::string, std::string> &set,
                       const pdb::PDBKeyExtractorPtr &keyExtractor);

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
  std::pair<bool, std::string> handleStoreData(const pdb::Handle<pdb::StoStoreDataRequest> &request,
                                               std::shared_ptr<Communicator> &sendUsingMe);

  /**
   *
   * @tparam Communicator
   * @param request
   * @param sendUsingMe
   * @return
   */
  template <class Communicator>
  std::pair<bool, std::string> handleStoreKeys(const pdb::Handle<pdb::StoStoreKeysRequest> &request,
                                               std::shared_ptr<Communicator> &sendUsingMe);


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
   * The index for a page set
   */
  map<std::pair<uint64_t, std::string>, TRAIndexNodePtr> pageSetsIndex;

  /**
   * the mutex to lock the page sets
   */
  std::mutex pageSetMutex;
};

using PDBStorageManagerBackendPtr = std::shared_ptr<PDBStorageManagerBackend>;

}

#include <PDBStorageManagerBackendTemplate.cc>