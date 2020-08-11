#pragma once

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
#include "StoDispatchKeys.h"


namespace pdb {


class PDBStorageManagerFrontend : public ServerFunctionality {

public:

  virtual ~PDBStorageManagerFrontend();

  /**
   * Initialize the storage manager frontend
   */
  void init() override;

  void registerHandlers(PDBServer &forMe) override;

  // This method contacts the frontend to get a PageSet for a particular PDB set
  PDBSetPageSetPtr createPageSetFromPDBSet(const std::string &db, const std::string &set, bool isKeyed);

  // this creates an anonymous page set
  PDBAnonymousPageSetPtr createAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID);

  // create a random access page set.
  PDBRandomAccessPageSetPtr createRandomAccessPageSet(const std::pair<uint64_t, std::string> &pageSetID);

  // create the feeding anonymous page set
  PDBFeedingPageSetPtr createFeedingAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID, uint64_t numReaders, uint64_t numFeeders);

  // this makes a page set that fetches pages of a given set from a particular node
  PDBAbstractPageSetPtr fetchPDBSet(const std::string &database, const std::string &set, bool isKey, const std::string &ip, int32_t port);

  // this makes a page se that fetches pages of a given page set from a particular node
  PDBAbstractPageSetPtr fetchPageSet(const PDBSourcePageSetSpec &pageSetSpec,
                                     bool isKey,
                                     const std::string &ip,
                                     int32_t port);

  // returns a pages set that already exists
  PDBAbstractPageSetPtr getPageSet(const std::pair<uint64_t, std::string> &pageSetID);

  // removes the page set from the storage.
  bool removePageSet(const std::pair<uint64_t, std::string> &pageSetID);

  // this method materializes a particular page set to a particular set. It contacts the frontend and grabs a bunch of pages
  // it assumes that the set we are materializing to exists.
  bool materializePageSet(const PDBAbstractPageSetPtr& pageSet, const std::pair<std::string, std::string> &set);

  // this materializes the keys of a particular page set
  bool materializeKeys(const PDBAbstractPageSetPtr& pageSet,
                       const std::pair<std::string, std::string> &set,
                       const pdb::PDBKeyExtractorPtr &keyExtractor);

 private:

  // this keeps track of the stats in a set
  struct PDBStorageSetStats {

    // the size of the set in bytes
    size_t size{0};

    // the number of records
    std::size_t numberOfRecords{0};

    // the number of keys
    std::size_t numberOfKeys{0};

    // the total size of the keys stored in the set
    std::size_t keysSize{0};

    // this tells us what the past page of the set is
    int64_t numberOfPages{0};

    // this tells us the number of key pages
    int64_t numberOfKeyPages{0};
  };

  // this handler basically accepts the data issued by the dispatcher onto a anonymous page,
  std::pair<bool, std::string> handleDispatchedData(const pdb::Handle<pdb::StoDispatchData>& request, const PDBCommunicatorPtr& sendUsingMe);

  // this handler accepts key data issued by the dispatches and stores it onto a page
  pair<bool, std::string> handleDispatchedKeys(const Handle<pdb::StoDispatchKeys>& request, const PDBCommunicatorPtr& sendUsingMe);

  // This is the response to @see requestPage. Basically it compresses the page and sends it's bytes over the wire to
  // the node that made the request.
  std::pair<bool, std::string> handleGetPageRequest(const pdb::Handle<pdb::StoGetPageRequest> &request, PDBCommunicatorPtr &sendUsingMe);

  // the logger
  PDBLoggerPtr logger;

  // This keeps track of the stats for a particular set. @see PDBStorageSetStats for the kind of information that is being stored
  map <PDBSetPtr, PDBStorageSetStats, PDBSetCompare> pageStats;

  // this locks the page stats generally used during the
  std::mutex pageStatsMutex;

};

// just make a shared ptr
using PDBStorageManagerFrontendPtr = std::shared_ptr<PDBStorageManagerFrontend>;

}

#include <PDBStorageManagerFrontendTemplate.cc>
