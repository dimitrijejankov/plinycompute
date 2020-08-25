//
// Created by dimitrije on 2/9/19.
//

#include <PDBStorageManager.h>
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
#include <StoFeedPageRequest.h>

namespace fs = boost::filesystem;

void pdb::PDBStorageManager::init() {

  // init the class
  logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "logs").string(),
                                       "PDBStorageManager.log");

  // do we have the metadata for the storage
  if (fs::exists(boost::filesystem::path(getConfiguration()->rootDirectory) / "storage.pdb")) {

    // open if stream
    std::ifstream ifs;
    ifs.open((boost::filesystem::path(getConfiguration()->rootDirectory) / "storage.pdb").string(),
             ios::binary | std::ifstream::in);

    size_t numSets;
    ifs.read((char *) &numSets, sizeof(unsigned long));

    for (int i = 0; i < numSets; ++i) {

      // read the database name
      unsigned long size;
      ifs.read((char *) &size, sizeof(unsigned long));
      std::unique_ptr<char[]> setBuffer(new char[size]);
      ifs.read(setBuffer.get(), size);
      std::string dbName(setBuffer.get(), size);

      // read the set name
      ifs.read((char *) &size, sizeof(unsigned long));
      std::unique_ptr<char[]> dbBuffer(new char[size]);
      ifs.read(dbBuffer.get(), size);
      std::string setName(dbBuffer.get(), size);

      // read the number of pages
      PDBStorageSetStats pageStat{};
      ifs.read(reinterpret_cast<char *>(&pageStat), sizeof(pageStat));

      // store the set info
      auto set = std::make_shared<PDBSet>(dbName, setName);
      this->pageStats[set] = pageStat;
    }

    // close
    ifs.close();
  }
}

void pdb::PDBStorageManager::cleanup() {

  // open the output file
  std::ofstream ofs;
  ofs.open((boost::filesystem::path(getConfiguration()->rootDirectory) / "storage.pdb").string(),
           ios::binary | std::ofstream::out | std::ofstream::trunc);

  unsigned long numSets = pageStats.size();
  ofs.write((char *) &numSets, sizeof(unsigned long));

  // serialize the stuff
  for (auto &it : pageStats) {

    // write the database name
    unsigned long size = it.first->getDBName().size();
    ofs.write((char *) &size, sizeof(unsigned long));
    ofs.write(it.first->getDBName().c_str(), size);

    // write the set name
    size = it.first->getSetName().size();
    ofs.write((char *) &size, sizeof(unsigned long));
    ofs.write(it.first->getSetName().c_str(), size);

    // write the page stats
    ofs.write(reinterpret_cast<char *>(&it.second), sizeof(it.second));
  }

  ofs.close();
}

void pdb::PDBStorageManager::registerHandlers(PDBServer &forMe) {

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
      StoRemovePageSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoRemovePageSetRequest>>([&](const Handle<pdb::StoRemovePageSetRequest>& request, const PDBCommunicatorPtr& sendUsingMe) {
        return handleRemovePageSet(request, sendUsingMe);
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
      make_shared<pdb::HeapRequestHandler<pdb::StoClearSetRequest>>([&](const Handle<pdb::StoClearSetRequest>& request, PDBCommunicatorPtr sendUsingMe) {
        return handleClearSetRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoStartFeedingPageSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoStartFeedingPageSetRequest>>(
          [&](const Handle<pdb::StoStartFeedingPageSetRequest>& request, const PDBCommunicatorPtr& sendUsingMe) {
            return handleStartFeedingPageSetRequest(request, sendUsingMe);
          }));

  forMe.registerHandler(
      StoFetchSetPagesRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoFetchSetPagesRequest>>([&](const Handle<pdb::StoFetchSetPagesRequest>& request, const PDBCommunicatorPtr& sendUsingMe) {
        return handleStoFetchSetPages(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoFetchPageSetPagesRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoFetchPageSetPagesRequest>>([&](const Handle<pdb::StoFetchPageSetPagesRequest>& request, const PDBCommunicatorPtr& sendUsingMe) {
        return handleStoFetchPageSetPagesRequest(request, sendUsingMe);
      }));
}

std::pair<bool, std::string> pdb::PDBStorageManager::handleDispatchedData(const pdb::Handle<pdb::StoDispatchData>& request,
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
    pdbClient.recordComMgr(*conMgr);
    if (!pdbClient.incrementSetRecordInfo(getConfiguration()->nodeID,
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

std::pair<bool, std::string> pdb::PDBStorageManager::handleDispatchedKeys(const pdb::Handle<pdb::StoDispatchKeys>& request,
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
    pdbClient.recordComMgr(*conMgr);
    if (!pdbClient.incrementKeyRecordInfo(getConfiguration()->nodeID,
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

std::pair<bool, std::string> pdb::PDBStorageManager::handleGetPageRequest(const pdb::Handle<pdb::StoGetPageRequest> &request,
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

std::pair<bool, std::string> pdb::PDBStorageManager::handleClearSetRequest(const pdb::Handle<pdb::StoClearSetRequest> &request,
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

std::pair<bool, std::string> pdb::PDBStorageManager::handleRemovePageSet(const pdb::Handle<pdb::StoRemovePageSetRequest> &request,
                                                                                 const pdb::PDBCommunicatorPtr &sendUsingMe) {

  /// 1. Remove the page set

  // remove the page set
  bool success = removePageSet(std::make_pair(request->pageSetID.first, request->pageSetID.second));

  // did we succeed in removing it?
  std::string error;
  if(!success) {
    error = "Could not find the page set " + std::to_string(request->pageSetID.first) + ":"+ std::string(request->pageSetID.second) + '\n';
  }

  /// 2. Send a response

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response
  pdb::Handle<pdb::SimpleRequestResult> simpleResponse = pdb::makeObject<pdb::SimpleRequestResult>(success, error);

  // sends result to requester
  sendUsingMe->sendObject(simpleResponse, error);

  // return
  return std::make_pair(success, error);
}

std::pair<bool, std::string> pdb::PDBStorageManager::handleStoFetchPageSetPagesRequest(const pdb::Handle<pdb::StoFetchPageSetPagesRequest> &request,
                                                                                       const shared_ptr<PDBCommunicator> &sendUsingMe) {
  throw runtime_error("Fetching pages from a page set is not yet supported");
}

std::pair<bool, std::string> pdb::PDBStorageManager::handleStoFetchSetPages(const pdb::Handle<pdb::StoFetchSetPagesRequest> &request,
                                                                            const shared_ptr<PDBCommunicator> &sendUsingMe) {
  /// 1. Get all the pages we need from the set

  // get the pages of the set and make the set identifier
  PDBSetPtr storageSet = nullptr;
  uint64_t numPages = 0;
  {
    // lock the stuff
    unique_lock<std::mutex> lck(pageSetMutex);

    // try to find the page
    auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);
    auto it = this->pageStats.find(set);
    if(it != this->pageStats.end()) {

      // check if we are requesting the keys or not
      if(!request->forKeys) {
        numPages = it->second.numberOfPages;

        // we are requesting a regular set
        storageSet = std::make_shared<PDBSet>(request->databaseName, request->setName);
      }
      else {
        numPages = it->second.numberOfKeyPages;

        // we request the key modify the set name
        storageSet = std::make_shared<PDBSet>(request->databaseName, PDBCatalog::toKeySetName(request->setName));
      }
    }
  }

  /// 2. Send a response with the number of pages to expect

  // make the allocation
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // make the response
  pdb::Handle<StoFetchPagesResponse> response = pdb::makeObject<StoFetchPagesResponse>(numPages);

  // send it
  std::string error;
  sendUsingMe->sendObject(response, error);

  // get the buffer manager
  auto bufferManager = getFunctionalityPtr<PDBBufferManagerInterface>();

  // make the response
  pdb::Handle<StoFetchNextPageResult> fetchNextPage = pdb::makeObject<StoFetchNextPageResult>();

  // go through each page
  for(auto page = 0; page < numPages; ++page) {

    // get the page handle
    auto pageHandle = bufferManager->getPage(storageSet, page);

    // set the page size and that we have another page
    fetchNextPage->pageSize = ((Record<Object> *) pageHandle->getBytes())->numBytes();
    fetchNextPage->hasNext = true;

    // send the fetch next page
    bool success = sendUsingMe->sendObject(fetchNextPage, error);

    // check if the
    if(!success) {

      // log the error
      logger->error(error);

      // we failed
      return std::make_pair(success, error);
    }

    // send the bytes
    success = sendUsingMe->sendBytes((char*) pageHandle->getBytes(), fetchNextPage->pageSize, error);

    // check if we failed to send the bytes
    if(!success) {

      // log the error
      logger->error(error);

      // we failed
      return std::make_pair(success, error);
    }
  }

  /// 3. Finish here

  // send the fetch next page
  fetchNextPage->hasNext = false;
  bool success = sendUsingMe->sendObject(fetchNextPage, error);

  // check if the
  if(!success) {

    // log the error
    logger->error(error);
  }

  // we failed
  return std::make_pair(success, error);
}

std::pair<bool, std::string> pdb::PDBStorageManager::handleStartFeedingPageSetRequest(const pdb::Handle<pdb::StoStartFeedingPageSetRequest> &request,
                                                                                      const pdb::PDBCommunicatorPtr &sendUsingMe) {
  bool success = true;
  std::string error;

  /// 1. Send a response to the other node about the status

  // create or grab the page set
  auto pageSet = createFeedingAnonymousPageSet(request->getPageSetID(), request->numberOfProcessingThreads, request->numberOfNodes);
  success = pageSet != nullptr;

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response for the other node
  pdb::Handle<pdb::SimpleRequestResult> simpleResponse = pdb::makeObject<pdb::SimpleRequestResult>(success, error);

  // sends result to requester
  success = sendUsingMe->sendObject(simpleResponse, error) && success;

  /// 2. If everything went well, start getting the pages from the other node

  // get the buffer manager
  auto bufferManager = getFunctionalityPtr<pdb::PDBBufferManagerInterface>();

  // if everything went well start receiving the pages
  while(success) {

    /// 3.1 Get the signal that there are more pages

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock localBlock{1024};

    // get the next object
    auto hasPage = sendUsingMe->template getNextObject<pdb::StoFeedPageRequest>(success, error);

    // if we failed break
    if(!success) {
      break;
    }

    // do we have a page, if we don't finish
    if(!hasPage->hasNextPage){
      break;
    }

    /// 3.2 Get the page from the other node

    // get the page of the size we need
    auto page = bufferManager->getPage(hasPage->pageSize);

    // grab the bytes
    success = sendUsingMe->receiveBytes(page->getBytes(), error);

    // if we failed finish
    if(!success) {
      break;
    }

    /// 3.3 Feed the page

    // unpin the page
    page->unpin();

    // feed the page to the page set
    pageSet->feedPage(page);
  }

  // if we got a page set mark that we are finished feeding
  if(pageSet != nullptr) {

    // finish feeding
    pageSet->finishFeeding();
  }

  // return
  return std::make_pair(success, error);
}

// page set stuff
pdb::PDBSetPageSetPtr pdb::PDBStorageManager::createPageSetFromPDBSet(const std::string &db,
                                                                      const std::string &set,
                                                                      bool requestingKeys) {


  // get the configuration
  auto conf = this->getConfiguration();

  /// 1. Get to get the number of pages

  // copy the stuff
  uint64_t numPages;
  {
    unique_lock<std::mutex> lck(this->pageStatsMutex);

    // make the set
    auto setPtr = std::make_shared<PDBSet>(db, set);

    // get the number of pages
    auto stats = pageStats.find(setPtr);
    if(stats == pageStats.end()) {
      // if we don't have any stats we assume that we have no pages
      numPages = 0;
    }
    else if(!requestingKeys) {
      // check if we are requesting keys
      numPages = stats->second.numberOfPages;
    }
    else {
      numPages = stats->second.numberOfKeyPages;
    }
  }

  // go through all the pages and set it to i
  // we do this since we might wanna extend the storage manager to skip pages that
  // might be written to or are not in a constant state, right now this is fine.
  std::vector<uint64_t> pages(numPages);
  for(auto i = 0; i < pages.size(); ++i) {
    pages[i] = i;
  }

  /// 2. Create it and return it

  if(!requestingKeys) {

    // store the page set
    return std::make_shared<pdb::PDBSetPageSet>(db, set, pages, getFunctionalityPtr<PDBBufferManagerInterface>());
  }
  else {

    // store the page set
    return std::make_shared<pdb::PDBSetPageSet>(db, PDBCatalog::toKeySetName(set), pages, getFunctionalityPtr<PDBBufferManagerInterface>());
  }
}

pdb::PDBAnonymousPageSetPtr pdb::PDBStorageManager::createAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID) {

  /// 1. Check if we already have the thing if we do return it

  std::unique_lock<std::mutex> lck(pageSetMutex);

  // try to find the page if it exists return it
  auto it = pageSets.find(pageSetID);
  if(it != pageSets.end()) {
    return std::dynamic_pointer_cast<PDBAnonymousPageSet>(it->second);
  }

  /// 2. We don't have it so create it

  // store the page set
  auto pageSet = std::make_shared<pdb::PDBAnonymousPageSet>(getFunctionalityPtr<PDBBufferManagerInterface>());
  pageSets[pageSetID] = pageSet;

  // return it
  return pageSet;
}

pdb::PDBRandomAccessPageSetPtr pdb::PDBStorageManager::createRandomAccessPageSet(const std::pair<uint64_t, std::string> &pageSetID) {

  /// 1. Check if we already have the thing if we do return it

  std::unique_lock<std::mutex> lck(pageSetMutex);

  // try to find the page if it exists return it
  auto it = pageSets.find(pageSetID);
  if(it != pageSets.end()) {
    return std::dynamic_pointer_cast<PDBRandomAccessPageSet>(it->second);
  }

  /// 2. We don't have it so create it

  // store the page set
  auto pageSet = std::make_shared<pdb::PDBRandomAccessPageSet>(getFunctionalityPtr<PDBBufferManagerInterface>());
  pageSets[pageSetID] = pageSet;

  // return it
  return pageSet;
}

pdb::PDBFeedingPageSetPtr pdb::PDBStorageManager::createFeedingAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID,
                                                                                        uint64_t numReaders,
                                                                                        uint64_t numFeeders) {
  /// 1. Check if we already have the thing if we do return it

  std::unique_lock<std::mutex> lck(pageSetMutex);

  // try to find the page if it exists return it
  auto it = pageSets.find(pageSetID);
  if(it != pageSets.end()) {
    return std::dynamic_pointer_cast<PDBFeedingPageSet>(it->second);
  }

  /// 2. We don't have it so create it

  // store the page set
  auto pageSet = std::make_shared<pdb::PDBFeedingPageSet>(numReaders, numFeeders);
  pageSets[pageSetID] = pageSet;

  // return it
  return pageSet;
}

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManager::fetchPDBSet(const std::string &database,
                                                                       const std::string &set,
                                                                       bool isKey,
                                                                       int32_t nodeID) {
  // get the configuration
  auto conf = this->getConfiguration();

  /// 1. Contact the node to establish a connection

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // the communicator
  string errMsg;
  PDBCommunicatorPtr comm = conMgr->connectTo(logger, nodeID, errMsg);

  // connect to the node
  if (comm == nullptr) {

    // log the error
    logger->error(errMsg);
    logger->error("Could not connect node " + std::to_string(nodeID) + " to fetch the set (" +  database + ":" + set + ").\n");

    // return null
    return nullptr;
  }

  // make request
  pdb::Handle<StoFetchSetPagesRequest> request = pdb::makeObject<StoFetchSetPagesRequest>(database, set, isKey);

  // send the object
  std::string error;
  bool success = comm->sendObject(request, error);

  // if we failed
  if(!success) {

    // log the error
    logger->error(error);

    // return a null pointer
    return nullptr;
  }

  /// 2. Grab the response

  // wait to get the number of pages
  uint64_t numPages;
  success = RequestFactory::waitHeapRequest<StoFetchPagesResponse, bool>(logger, comm, false,
[&](const Handle<StoFetchPagesResponse>& result) {

   // check the result
   if (result != nullptr) {
     numPages = result->numPages;
     return true;
   }

   // log the error
   error = "Error getting the number of pages for the fetching page set!";
   logger->error(error);

   return false;
  });

  // did we fail if we did return null
  if(!success) {
    return nullptr;
  }

  /// 3. Create the page set since we are about to receive the pages

  auto storageManager = getFunctionalityPtr<PDBStorageManager>();
  auto bufferManager = getFunctionalityPtr<PDBBufferManagerInterface>();

  // return the fetching page set
  return std::make_shared<pdb::PDBFetchingPageSet>(comm,
                                                   storageManager,
                                                   bufferManager,
                                                   numPages);
}

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManager::fetchPageSet(const pdb::PDBSourcePageSetSpec &pageSetSpec,
                                                                bool isKey,
                                                                int32_t nodeID) {
  // get the configuration
  auto conf = this->getConfiguration();

  /// 1. Contact the node to establish a connection

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // the communicator
  string errMsg;
  PDBCommunicatorPtr comm = conMgr->connectTo(logger, nodeID, errMsg);

  // connect to the node
  if (comm == nullptr) {

    // log the error
    logger->error(errMsg);
    logger->error("Could not connect node " + std::to_string(nodeID) +
        " to fetch the page set (" +  std::to_string(pageSetSpec.pageSetIdentifier.first) +
        ":" + (std::string)pageSetSpec.pageSetIdentifier.second + ").\n");

    // return null
    return nullptr;
  }

  // make request
  pdb::Handle<StoFetchPageSetPagesRequest> request = pdb::makeObject<StoFetchPageSetPagesRequest>(pageSetSpec.pageSetIdentifier, isKey);

  // send the object
  std::string error;
  bool success = comm->sendObject(request, error);

  // if we failed
  if(!success) {

    // log the error
    logger->error(error);

    // return a null pointer
    return nullptr;
  }

  /// 2. Grab the response

  // wait to get the number of pages
  uint64_t numPages;
  success = RequestFactory::waitHeapRequest<StoFetchPagesResponse, bool>(logger, comm, false,
                                                                         [&](const Handle<StoFetchPagesResponse>& result) {

                                                                           // check the result
                                                                           if (result != nullptr) {
                                                                             numPages = result->numPages;
                                                                             return true;
                                                                           }

                                                                           // log the error
                                                                           error = "Error getting the number of pages for the fetching page set!";
                                                                           logger->error(error);

                                                                           return false;
                                                                         });

  // did we fail if we did return null
  if(!success) {
    return nullptr;
  }

  /// 3. Create the page set since we are about to receive the pages

  // return the fetching page set
  return std::make_shared<pdb::PDBFetchingPageSet>(comm,
                                                   getFunctionalityPtr<PDBStorageManager>(),
                                                   getFunctionalityPtr<PDBBufferManagerInterface>(),
                                                   numPages);
}

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManager::getPageSet(const std::pair<uint64_t,
                                                                                      std::string> &pageSetID) {
  // try to find the page if it exists return it
  auto it = pageSets.find(pageSetID);
  if(it != pageSets.end()) {
    return std::dynamic_pointer_cast<PDBAbstractPageSet>(it->second);
  }

  // return null since we don't have it
  return nullptr;
}

bool pdb::PDBStorageManager::removePageSet(const std::pair<uint64_t, std::string> &pageSetID) {

  // erase it if it exists
  return pageSets.erase(pageSetID) == 1;
}

bool pdb::PDBStorageManager::materializePageSet(const pdb::PDBAbstractPageSetPtr &pageSet,
                                                const std::pair<std::string, std::string> &set) {

  // if the page set is empty no need materialize stuff
  if(pageSet->getNumPages() == 0) {
    return true;
  }

  // result indicators
  std::string error;

  // buffer manager
  auto numPages = pageSet->getNumPages();
  auto bufferManager = getFunctionalityPtr<pdb::PDBBufferManagerInterface>();
  auto setIdentifier = std::make_shared<PDBSet>(set.first, set.second);

  {
    // locks the page stats structure
    unique_lock<std::mutex> lck(this->pageStatsMutex);

    // update the stats
    auto &stats = pageStats[setIdentifier];

    // if we have anything in this set, we can not materialize
    if(!stats.empty()) {
      return false;
    }

    stats.numberOfPages = numPages;
    stats.numberOfRecords = pageSet->getNumRecords();
    stats.size = pageSet->getSize();
  }

  // go through each page and materialize
  PDBPageHandle page;
  for (int i = 0; i < numPages; ++i) {

    // grab the next page and move it to the set
    page = pageSet->getNextPage(0);

    // repin this page
    page->repin();

    // move it
    page->move(setIdentifier, i);
  }

  /// 4. Update the set size

  // broadcast the set size change so far
  PDBCatalogClient pdbClient(getConfiguration()->managerPort, getConfiguration()->managerAddress, logger);
  pdbClient.recordComMgr(*conMgr);
  bool success = pdbClient.incrementSetRecordInfo(getConfiguration()->nodeID, set.first, set.second,
                                                  pageSet->getSize(), pageSet->getNumRecords(), error);


  return success;
}

bool pdb::PDBStorageManager::materializeKeys(const pdb::PDBAbstractPageSetPtr &pageSet,
                                             const std::pair<std::string, std::string> &set,
                                             const pdb::PDBKeyExtractorPtr &keyExtractor) {

  // make the key set identifier
  auto keySetIdentifier = std::make_shared<PDBSet>(set.first, PDBCatalog::toKeySetName(set.second));
  auto setIdentifier = std::make_shared<PDBSet>(set.first, set.second);

  {
    // locks the page stats structure
    unique_lock<std::mutex> lck(this->pageStatsMutex);

    // update the stats
    auto &stats = pageStats[setIdentifier];

    // if we have anything in this set, we can not materialize
    if (!stats.keysEmpty()) {
      return false;
    }
  }

  // if the page set is empty no need materialize stuff
  if(pageSet->getNumPages() == 0) {
    return true;
  }

  // get the number of pages
  auto numPages = pageSet->getNumPages();
  auto bufferManager = getFunctionalityPtr<pdb::PDBBufferManagerInterface>();

  // grab a page
  auto keyPage = bufferManager->getPage(keySetIdentifier, 0);

  // make the allocation block
  makeObjectAllocatorBlock(keyPage->getBytes(), keyPage->getSize(), true);

  // while we still have pages and we have processed last one
  PDBPageHandle inputPage;
  while(numPages != 0) {

    // grab the next page
    inputPage = pageSet->getNextPage(0);

    // repin the page
    inputPage->repin();

    // copy the memory to the set page
    try {

      // extract the keys from the page
      keyExtractor->extractKeys(inputPage, keyPage);
    }
    catch (pdb::NotEnoughSpace &n) {

      // TODO we only support a single key page
      return false;
    }

    // decrement the number of pages
    numPages--;
  }

  uint64_t keysSize = 0;
  uint64_t numberOfKeys = 0;
  {
    // locks the page stats structure
    unique_lock<std::mutex> lck(this->pageStatsMutex);

    // update the stats
    auto &stats = pageStats[setIdentifier];

    keysSize = keyExtractor->pageSize(keyPage);
    numberOfKeys = keyExtractor->numTuples(keyPage);

    // update the key info
    stats.numberOfKeyPages = 1;
    stats.keysSize = keysSize;
    stats.numberOfKeys = numberOfKeys;
  }

  // move the page to the set
  //keyPage->move(keySetIdentifier, 0);

  // make an allocation block to send the response
  pdb::makeObjectAllocatorBlock(1024, true);

  // broadcast the set size change so far
  std::string error;
  PDBCatalogClient pdbClient(getConfiguration()->managerPort, getConfiguration()->managerAddress, logger);
  pdbClient.recordComMgr(*conMgr);
  bool success = pdbClient.incrementKeyRecordInfo(getConfiguration()->nodeID,
                                                  setIdentifier->getDBName(),
                                                  setIdentifier->getSetName(),
                                                  keysSize,
                                                  numberOfKeys,
                                                  error);


  // we succeeded yay!
  return success;
}
