//
// Created by dimitrije on 2/9/19.
//

#include <PDBStorageManagerFrontend.h>
#include <HeapRequestHandler.h>
#include <StoDispatchData.h>
#include <PDBBufferManagerInterface.h>
#include <PDBBufferManagerImpl.h>
#include <StoStoreDataRequest.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <fstream>
#include <utility>
#include <HeapRequest.h>
#include <StoGetNextPageRequest.h>
#include <StoGetNextPageResult.h>
#include "CatalogServer.h"
#include <StoGetPageRequest.h>
#include <StoGetPageResult.h>
#include <StoMaterializePageSetRequest.h>
#include <StoStartFeedingPageSetRequest.h>
#include <StoDispatchKeys.h>
#include <StoStoreKeysRequest.h>
#include <StoFetchPagesResponse.h>
#include <StoFetchNextPageResult.h>
#include <StoMaterializeKeysRequest.h>
#include <PDBFetchingPageSet.h>

namespace fs = boost::filesystem;

pdb::PDBStorageManagerFrontend::~PDBStorageManagerFrontend() = default;

void pdb::PDBStorageManagerFrontend::init() {

  // init the class
  logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "logs").string(),
                                       "PDBStorageManager.log");
}

void pdb::PDBStorageManagerFrontend::registerHandlers(PDBServer &forMe) {

  forMe.registerHandler(
      StoDispatchData_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoDispatchData>>(
          [&](const Handle<pdb::StoDispatchData>& request, const PDBCommunicatorPtr& sendUsingMe) {
            return handleDispatchedData(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoDispatchKeys_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoDispatchKeys>>(
          [&](const Handle<pdb::StoDispatchKeys>& request, const PDBCommunicatorPtr& sendUsingMe) {
            return handleDispatchedKeys(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoGetPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoGetPageRequest>>([&](const Handle<pdb::StoGetPageRequest>& request,
                                                                                        PDBCommunicatorPtr sendUsingMe) {
        // handle the get page request
        return handleGetPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoClearSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoClearSetRequest>>([&](Handle<pdb::StoClearSetRequest> request, PDBCommunicatorPtr sendUsingMe) {
        return handleClearSetRequest(request, sendUsingMe);
      }));
}

std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleDispatchedData(const pdb::Handle<pdb::StoDispatchData>& request,
                                                                                  const pdb::PDBCommunicatorPtr& sendUsingMe) {

  /// 1. Get the page from the distributed storage

  // the error
  std::string error;

  // grab the buffer manager
  auto bufferManager = getFunctionalityPtr<pdb::PDBBufferManagerInterface>();

  // figure out how large the compressed payload is
  size_t numBytes = sendUsingMe->getSizeOfNextObject();

  // grab a page to write this
  auto inputPage = bufferManager->getPage(numBytes);

  // grab the bytes
  auto success = sendUsingMe->receiveBytes(inputPage->getBytes(), error);

  // did we fail
  if(!success) {

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};
    Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, error);

    // sends result to requester
    sendUsingMe->sendObject(response, error);

    return std::make_pair(false, error);
  }

  // figure out the size so we can increment it
  // check the uncompressed size
  size_t uncompressedSize = 0;
  snappy::GetUncompressedLength((char*) inputPage->getBytes(), numBytes, &uncompressedSize);

  /// 2. Figure out the page we want to put this thing onto

  // make the set
  auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

  // fetch a new page number
  int64_t newPageNumber;
  {
    unique_lock<std::mutex> lck(this->pageStatsMutex);

    // get the stats for this set get a new number
    auto &stats = pageStats[set];
    newPageNumber = stats.numberOfPages++;
  }

  // get the new page
  auto newPage = bufferManager->getPage(set, newPageNumber);

  // uncompress and copy to page
  snappy::RawUncompress((char*) inputPage->getBytes(), request->compressedSize, (char*) newPage->getBytes());

  // freeze the page
  newPage->freezeSize(uncompressedSize);

  /// 3. Update the set size
  {
    // figure out the number of records
    Handle<Vector<Handle<Object>>> data = ((Record<Vector<Handle<Object>>> *) newPage->getBytes())->getRootObject();
    uint64_t numRecords = data->size();

    // increment the number of records
    {
      unique_lock<std::mutex> lck(this->pageStatsMutex);

      // get the stats for this set and update them
      auto &stats = pageStats[set];
      stats.size += ((Record<Vector<Handle<Object>>> *) newPage->getBytes())->numBytes();
      stats.numberOfRecords += numRecords;
    }

    // send the catalog that data has been added
    std::string errMsg;
    PDBCatalogClient pdbClient(getConfiguration()->managerPort, getConfiguration()->managerAddress, logger);
    if (!pdbClient.incrementSetRecordInfo(getConfiguration()->getNodeIdentifier(),
                                          request->databaseName,
                                          request->setName,
                                          uncompressedSize,
                                          numRecords,
                                          errMsg)) {

      // create an allocation block to hold the response
      const UseTemporaryAllocationBlock tempBlock{1024};
      Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

      // sends result to requester
      sendUsingMe->sendObject(response, errMsg);
      return make_pair(false, errMsg);
    }
  }

  /// 4. Send the response that we are done

  // create an allocation block to hold the response
  Handle<SimpleRequestResult> simpleResponse = makeObject<SimpleRequestResult>(success, error);

  // sends result to requester
  success = sendUsingMe->sendObject(simpleResponse, error) && success;

  // finish
  return std::make_pair(success, error);
}

std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleDispatchedKeys(const pdb::Handle<pdb::StoDispatchKeys>& request,
                                                                                  const pdb::PDBCommunicatorPtr& sendUsingMe) {
  /// 1. Get the page from the distributed storage

  // the error
  std::string error;

  // grab the buffer manager
  auto bufferManager = getFunctionalityPtr<pdb::PDBBufferManagerInterface>();

  // figure out how large the compressed payload is
  size_t numBytes = sendUsingMe->getSizeOfNextObject();

  // grab a page to write this
  auto inputPage = bufferManager->getPage(numBytes);

  // grab the bytes
  auto success = sendUsingMe->receiveBytes(inputPage->getBytes(), error);

  // did we fail
  if(!success) {

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};
    Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, error);

    // sends result to requester
    sendUsingMe->sendObject(response, error);

    return std::make_pair(false, error);
  }

  // figure out the size so we can increment it
  // check the uncompressed size
  size_t uncompressedSize = 0;
  snappy::GetUncompressedLength((char*) inputPage->getBytes(), numBytes, &uncompressedSize);

  /// 2. Figure out the page we want to put this thing onto

  // make the set
  auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);
  auto keySet = std::make_shared<PDBSet>(request->databaseName, PDBCatalog::toKeySetName(std::string(request->setName)));

  // fetch a new page number
  int64_t newPageNumber;
  {
    unique_lock<std::mutex> lck(this->pageStatsMutex);

    // get the stats for this set get a new number
    auto &stats = pageStats[set];
    newPageNumber = stats.numberOfKeyPages++;
  }

  // get the new page
  auto newPage = bufferManager->getPage(keySet, newPageNumber);

  // uncompress and copy to page
  snappy::RawUncompress((char*) inputPage->getBytes(), request->compressedSize, (char*) newPage->getBytes());

  // freeze the page
  newPage->freezeSize(uncompressedSize);

  /// 3. Update the set info about the keys
  {
    // figure out the number of records
    Handle<Vector<Handle<Object>>> data = ((Record<Vector<Handle<Object>>> *) newPage->getBytes())->getRootObject();
    uint64_t numRecords = data->size();

    // increment the number of records
    {
      unique_lock<std::mutex> lck(this->pageStatsMutex);

      // get the stats for this set and update them
      auto &stats = pageStats[set];
      stats.keysSize += ((Record<Vector<Handle<Object>>> *) newPage->getBytes())->numBytes();
      stats.numberOfKeys += numRecords;
    }

    // send the catalog that data has been added
    std::string errMsg;
    PDBCatalogClient pdbClient(getConfiguration()->managerPort, getConfiguration()->managerAddress, logger);
    if (!pdbClient.incrementKeyRecordInfo(getConfiguration()->getNodeIdentifier(),
                                          request->databaseName,
                                          request->setName,
                                          ((Record<Vector<Handle<Object>>> *) newPage->getBytes())->numBytes(),
                                          numRecords,
                                          errMsg)) {

      // create an allocation block to hold the response
      const UseTemporaryAllocationBlock tempBlock{1024};
      Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

      // sends result to requester
      sendUsingMe->sendObject(response, errMsg);
      return make_pair(false, errMsg);
    }
  }

  /// 4. Send the response that we are done

  // create an allocation block to hold the response
  Handle<SimpleRequestResult> simpleResponse = makeObject<SimpleRequestResult>(success, error);

  // sends result to requester
  success = sendUsingMe->sendObject(simpleResponse, error) && success;

  // finish
  return std::make_pair(success, error);
}

std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleGetPageRequest(const pdb::Handle<pdb::StoGetPageRequest> &request,
                                                                                  pdb::PDBCommunicatorPtr &sendUsingMe) {
  /// 1. Check if we have a page

  // create the set identifier
  auto set = make_shared<pdb::PDBSet>(request->databaseName, request->setName);

  // check if whe have the page
  bool hasPage;
  {
    unique_lock<std::mutex> lck(this->pageStatsMutex);

    // check if
    auto &stats = pageStats[set];
    hasPage = request->page < stats.numberOfPages;
  }

  /// 2. If we don't have it or it is not in a valid state it send back a NACK

  if(!hasPage) {

    // make an allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{1024};

    // create an allocation block to hold the response
    pdb::Handle<pdb::StoGetPageResult> response = pdb::makeObject<pdb::StoGetPageResult>(0, 0, false);

    // sends result to requester
    string error;
    sendUsingMe->sendObject(response, error);

    // This is an issue we simply return false only a manager can serve pages
    return make_pair(false, error);
  }

  /// 3. Ok we have it, grab the page and compress it.

  // grab the page
  auto page = this->getFunctionalityPtr<PDBBufferManagerInterface>()->getPage(set, request->page);

  // grab the vector
  auto* pageRecord = (pdb::Record<pdb::Vector<pdb::Handle<pdb::Object>>> *) (page->getBytes());

  // grab an anonymous page to store the compressed stuff //TODO this kind of sucks since the max compressed size can be larger than the actual size
  auto maxCompressedSize = std::min<size_t>(snappy::MaxCompressedLength(pageRecord->numBytes()), getFunctionalityPtr<PDBBufferManagerInterface>()->getMaxPageSize());
  auto compressedPage = getFunctionalityPtr<PDBBufferManagerInterface>()->getPage(maxCompressedSize);

  // compress the record
  size_t compressedSize;
  snappy::RawCompress((char*) pageRecord, pageRecord->numBytes(), (char*) compressedPage->getBytes(), &compressedSize);

  /// 4. Send the compressed page

  // make an allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{1024};

  // create an allocation block to hold the response
  pdb::Handle<pdb::StoGetPageResult> response = pdb::makeObject<pdb::StoGetPageResult>(compressedSize, request->page, true);

  // sends result to requester
  string error;
  sendUsingMe->sendObject(response, error);

  // now, send the bytes
  if (!sendUsingMe->sendBytes(compressedPage->getBytes(), compressedSize, error)) {

    this->logger->error(error);
    this->logger->error("sending page bytes: not able to send data to client.\n");

    // we are done here
    return make_pair(false, string(error));
  }

  // we are done!
  return make_pair(true, string(""));
}

std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleClearSetRequest(const pdb::Handle<pdb::StoClearSetRequest> &request,
                                                                                   pdb::PDBCommunicatorPtr &sendUsingMe) {
  std::string error;
  PDBSetPtr set;
  PDBSetPtr keySet;
  {
    // lock the structures
    unique_lock<std::mutex> lck(this->pageStatsMutex);

    // make the set
    set = std::make_shared<PDBSet>(request->databaseName, request->setName);

    // if we have keys we need to clear them too
    if(pageStats[set].numberOfKeys > 0) {
      keySet = std::make_shared<PDBSet>(request->databaseName, PDBCatalog::toKeySetName(std::string(request->setName)));
    }

    // remove the stats
    pageStats.erase(set);
  }

  // get the buffer manger
  auto bufferManager = getFunctionalityPtr<pdb::PDBBufferManagerInterface>();

  // clear the set from the buffer manager
  bufferManager->clearSet(set);

  // clear the keys if we have them too...
  if(keySet != nullptr) {
    bufferManager->clearSet(keySet);
  }

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};
  Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(true, error);

  // sends result to requester
  sendUsingMe->sendObject(response, error);

  // return
  return std::make_pair(true, error);
}

// page set stuff
pdb::PDBSetPageSetPtr pdb::PDBStorageManagerFrontend::createPageSetFromPDBSet(const std::string &db,
                                                                              const std::string &set,
                                                                              bool isKeyed) {
  return nullptr;
}

pdb::PDBAnonymousPageSetPtr pdb::PDBStorageManagerFrontend::createAnonymousPageSet(const std::pair<uint64_t,
                                                                                                   std::string> &pageSetID) {
  return nullptr;
}

pdb::PDBRandomAccessPageSetPtr pdb::PDBStorageManagerFrontend::createRandomAccessPageSet(const std::pair<uint64_t,
                                                                                                         std::string> &pageSetID) {
  return nullptr;
}

pdb::PDBFeedingPageSetPtr pdb::PDBStorageManagerFrontend::createFeedingAnonymousPageSet(const std::pair<uint64_t,
                                                                                                        std::string> &pageSetID,
                                                                                        uint64_t numReaders,
                                                                                        uint64_t numFeeders) {
  return nullptr;
}

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManagerFrontend::fetchPDBSet(const std::string &database,
                                                                       const std::string &set,
                                                                       bool isKey,
                                                                       const std::string &ip,
                                                                       int32_t port) {
  return nullptr;
}

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManagerFrontend::fetchPageSet(const pdb::PDBSourcePageSetSpec &pageSetSpec,
                                                                        bool isKey,
                                                                        const std::string &ip,
                                                                        int32_t port) {
  return nullptr;
}

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManagerFrontend::getPageSet(const std::pair<uint64_t,
                                                                                      std::string> &pageSetID) {
  return nullptr;
}

bool pdb::PDBStorageManagerFrontend::removePageSet(const std::pair<uint64_t, std::string> &pageSetID) {
  return false;
}

bool pdb::PDBStorageManagerFrontend::materializePageSet(const pdb::PDBAbstractPageSetPtr &pageSet,
                                                        const std::pair<std::string, std::string> &set) {
  return false;
}

bool pdb::PDBStorageManagerFrontend::materializeKeys(const pdb::PDBAbstractPageSetPtr &pageSet,
                                                     const std::pair<std::string, std::string> &set,
                                                     const pdb::PDBKeyExtractorPtr &keyExtractor) {
  return false;
}


