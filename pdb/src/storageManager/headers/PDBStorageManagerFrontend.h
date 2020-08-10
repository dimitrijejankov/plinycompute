#ifndef PDB_STORAGEMANAGERFRONTEND_H
#define PDB_STORAGEMANAGERFRONTEND_H

#include <mutex>
#include <unordered_set>

#include <PDBSet.h>
#include <PDBPageCompare.h>
#include <PDBCatalogNode.h>
#include <ServerFunctionality.h>
#include <PDBPageHandle.h>
#include <StoGetPageRequest.h>
#include <StoGetNextPageRequest.h>
#include <StoDispatchData.h>
#include <StoGetSetPagesRequest.h>
#include <StoMaterializePageSetRequest.h>
#include <StoMaterializePageResult.h>
#include <StoRemovePageSetRequest.h>
#include <StoStartFeedingPageSetRequest.h>
#include <StoClearSetRequest.h>
#include <StoFetchSetPagesRequest.h>
#include <StoFetchPageSetPagesRequest.h>
#include <StoMaterializeKeysRequest.h>
#include <StoStoreKeysRequest.h>
#include <PDBAnonymousPageSet.h>
#include <PDBRandomAccessPageSet.h>
#include <PDBFeedingPageSet.h>
#include "PDBSetPageSet.h"
#include <StoStoreDataRequest.h>
#include <PDBKeyExtractor.h>
#include <physicalAlgorithms/PDBSourcePageSetSpec.h>
#include "PDBAbstractPageSet.h"
#include "StoRemovePageSetRequest.h"
#include "StoStartFeedingPageSetRequest.h"




namespace pdb {

struct PDBStorageSetStats {

  /**
   * The size of the set
   */
  size_t size;

  /**
   * The number of pages
   */
  size_t lastPage;

};

class PDBStorageManagerFrontend : public ServerFunctionality {

public:

  virtual ~PDBStorageManagerFrontend();

  /**
   * Initialize the storage manager frontend
   */
  void init() override;

  void registerHandlers(PDBServer &forMe) override;

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
  template <class Communicator, class Requests, class RequestType, class ForwardRequestType>
  std::pair<bool, std::string> handleDispatchedData(pdb::Handle<RequestType> request, std::shared_ptr<Communicator> sendUsingMe);


  /**
   * Handles the the request to get stats about a particular set.
   *
   * @tparam Communicator- the communicator class PDBCommunicator is used to handle the request. This is basically here
   * so we could write unit tests
   *
   * @tparam Requests - the factory class to make request. RequestsFactory class is being used this is just here as a template so we
   * can mock it in the unit tests
   *
   * @param request - request to get the stats of particular set within this nodes storage. Contains the database and set names
   * @param sendUsingMe - the communicator to the node that made the request
   * @return - the result of the handler (success, error)
   */
  template <class Communicator, class Requests>
  std::pair<bool, std::string> handleGetSetPages(pdb::Handle<pdb::StoGetSetPagesRequest> request,
                                                 std::shared_ptr<Communicator> sendUsingMe);

  /**
   * Handles the materialization request of the backend. Basically it forwards a bunch of pages to the backend and check whether the
   * materialization is successful
   *
   * @tparam Communicator - the communicator class
   * @tparam Requests - the request factor class
   * @param request - the materialization request has stuff like the number of pages required for materialization etc..
   * @param sendUsingMe - the communicator to the backend
   * @return - the result of the handler (success, error)
   */
  template <class Communicator, class Requests>
  std::pair<bool, std::string> handleMaterializeSet(const pdb::Handle<pdb::StoMaterializePageSetRequest>& request, std::shared_ptr<Communicator> sendUsingMe);

  /**
   * Handles the materialization of keys the backend has issued the backend. Basically it forwards a bunch of pages to the backend and check whether the
   * materialization is successful
   *
   * @tparam Communicator - the communicator class
   * @tparam Requests - the request factor class
   * @param request - the materialization request has stuff like the number of pages required for materialization etc..
   * @param sendUsingMe - the communicator to the backend
   * @return - the result of the handler (success, error)
   */
  template <class Communicator, class Requests>
  std::pair<bool, std::string> handleMaterializeKeysSet(const pdb::Handle<pdb::StoMaterializeKeysRequest>& request, std::shared_ptr<Communicator> sendUsingMe);

  /**
   * Handles the request to remove a page set. It basically forwards that request to the backend.
   * @tparam Communicator - the communicator class
   *
   * @param request - the request, contains the info about the page set we want to remove
   * @param sendUsingMe - the communicator
   * @return - the result of the handler (success, error)
   */
  template <class Communicator>
  std::pair<bool, std::string> handleRemovePageSet(pdb::Handle<pdb::StoRemovePageSetRequest> &request, std::shared_ptr<Communicator> &sendUsingMe);

  /**
   *
   * @tparam Communicator
   * @param request
   * @param sendUsingMe
   * @return
   */
  /// template <class Communicator>
  ///TODO std::pair<bool, std::string> handleStartFeedingPageSetRequest(pdb::Handle<pdb::StoStartFeedingPageSetRequest> &request, std::shared_ptr<Communicator> &sendUsingMe);

  /**
   *
   * @tparam Communicator
   * @param request
   * @param sendUsingMe
   * @return
   */
  template <class Communicator>
  std::pair<bool, std::string> handleClearSetRequest(pdb::Handle<pdb::StoClearSetRequest> &request,
                                                     std::shared_ptr<Communicator> &sendUsingMe);


  /**
   *
   * @param request
   * @param sendUsingMe
   * @return
   */
  std::pair<bool, std::string> handleStoFetchSetPages(pdb::Handle<pdb::StoFetchSetPagesRequest> &request,
                                                      std::shared_ptr<PDBCommunicator> &sendUsingMe);

  /**
   *
   * @param request
   * @param sendUsingMe
   * @return
   */
  std::pair<bool, std::string> handleStoFetchPageSetPagesRequest(pdb::Handle<pdb::StoFetchPageSetPagesRequest> &request,
                                                                 std::shared_ptr<PDBCommunicator> &sendUsingMe);

  /**
   * This method handles the situation where we want to reclaim a page of a set that was allocated for the backend to
   * put the dispatched data to. We want to call this in case some unpredicted error happens
   * This method is thread safe so no locking required!
   *
   * @param set - the set of the page
   * @param pageNum - the page number
   * @param size - the size of the page
   * @param communicator - the communicator to send a NACK to the disptacher
   * @return true if it succeeds false if it fails
   */
  bool handleDispatchFailure(const PDBSetPtr &set, uint64_t pageNum, uint64_t size, const PDBCommunicatorPtr& communicator);

  /**
   * Checks whether we are writing to a particular page.
   * This method is not thread-safe and should only be used when locking the page mutex
   *
   * @param set - the name of the set the page we are checking belongs to
   * @param pageNum - the page number
   * @return - true if we are writing to that page false otherwise
   */
  bool isPageBeingWrittenTo(const PDBSetPtr &set, uint64_t pageNum);

  /**
   * Check whether the page is free for some reason.
   * This method is not thread-safe and should only be used when locking the page mutex
   * @param set - the name of the set the page we are checking belongs to
   * @param pageNum - the page number
   * @return true if is free, false otherwise
   */
  bool isPageFree(const PDBSetPtr &set, uint64_t pageNum);

  /**
   * Checks whether the page exists, no mater what state it is in. For example one could be writing currently to it
   * or it could be a free page.
   * This method is not thread-safe and should only be used when locking the page mutex
   *
   * @param set - the name of the set the page we are checking belongs to
   * @param pageNum - the page number
   * @return true if does false otherwise
   */
  bool pageExists(const PDBSetPtr &set, uint64_t pageNum);

  /**
   * Returns this page or the next page that has data on it and is not being written to, if such page exists
   * This is thread safe to call
   * @param set - the set
   * @param pageNum - the page number
   * @return the pair <hasPage, pageNumber>
   */
  std::pair<bool, uint64_t> getValidPage(const PDBSetPtr &set, uint64_t pageNum);

  /**
   * This method returns the next free page it can find.
   * If there are free pages in the @see freeSkippedPages then we will use those otherwise we will get the next page
   * after the last page
   * This method is not thread-safe and should only be used when locking the page mutex
   *
   * @param set - the set we want to get the free page for
   * @return the id of the next page.
   */
  uint64_t getNextFreePage(const PDBSetPtr &set);

  /**
   * this method marks a page as free, meaning, that it can be assigned by get @see getNextFreePage
   * This method is not thread-safe and should only be used when locking the page mutex
   *
   * @param set - the set the page belongs to
   * @param pageNum - the number of that page
   */
  void freeSetPage(const PDBSetPtr &set, uint64_t pageNum);

  /**
   * Mark the page as being written to so that it can not be sent
   * This method is not thread-safe and should only be used when locking the page mutex
   *
   * @param set - the set the page belongs to
   * @param pageNum - the number of that page
   */
  void startWritingToPage(const PDBSetPtr &set, uint64_t pageNum);

  /**
   * Unmark the page as being written to so that the storage can send it for reading and stuff
   * This method is not thread-safe and should only be used when locking the page mutex
   *
   * @param set - the set the page belongs to
   * @param pageNum - the number of that page
   */
  void endWritingToPage(const PDBSetPtr &set, uint64_t pageNum);

  /**
   * This method increments the set size. It assumes the set exists, should not be called unless it exists!
   * This method is not thread-safe and should only be used when locking the page mutex
   * @param set
   */
  void incrementSetSize(const PDBSetPtr &set, uint64_t uncompressedSize);

  /**
   * This method decrements the set size. It assumes the set exists, should not be called unless it exists!
   * This method is not thread-safe and should only be used when locking the page mutex
   * @param set
   */
  void decrementSetSize(const PDBSetPtr &set, uint64_t uncompressedSize);

  /**
   * Retu
   *
   * @param set
   * @return
   */
  std::shared_ptr<PDBStorageSetStats> getSetStats(const PDBSetPtr &set);

  /**
   * The logger
   */
  PDBLoggerPtr logger;

  /**
   * This keeps track of the stats for a particular set. @see PDBStorageSetStats for the kind of information that is being stored
   */
  map <PDBSetPtr, PDBStorageSetStats, PDBSetCompare> pageStats;

  /**
   * Pages that are currently being written to for a particular set
   */
  map<PDBSetPtr, std::unordered_set<uint64_t>, PDBSetCompare> pagesBeingWrittenTo;

  /**
   * The pages that we skipped for some reason when writing to. This can happen when some requests fail or something of that sort.
   */
  map<PDBSetPtr, std::unordered_set<uint64_t>, PDBSetCompare> freeSkippedPages;

  /**
   * Lock last pages
   */
  std::mutex pageMutex;


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
   * the mutex to lock the page sets
   */
  std::mutex pageSetMutex;
};

// just make a shared ptr
using PDBStorageManagerFrontendPtr = std::shared_ptr<PDBStorageManagerFrontend>;

}

#include <PDBStorageManagerFrontendTemplate.cc>

#endif //PDB_STORAGEMANAGERFRONTEND_H
