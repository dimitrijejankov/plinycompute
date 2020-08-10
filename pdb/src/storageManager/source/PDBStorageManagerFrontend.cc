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

pdb::PDBStorageManagerFrontend::~PDBStorageManagerFrontend() {

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

void pdb::PDBStorageManagerFrontend::init() {

  // init the class
  logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "logs").string(),
                                       "PDBStorageManagerFrontend.log");

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

void pdb::PDBStorageManagerFrontend::registerHandlers(PDBServer &forMe) {

  forMe.registerHandler(
      StoGetPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoGetPageRequest>>([&](const Handle<pdb::StoGetPageRequest>& request,
                                                                       PDBCommunicatorPtr sendUsingMe) {
        // handle the get page request
        return handleGetPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoDispatchData_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoDispatchData>>(
          [&](const Handle<pdb::StoDispatchData>& request, PDBCommunicatorPtr sendUsingMe) {
            return handleDispatchedData<PDBCommunicator, RequestFactory, pdb::StoDispatchData, StoStoreDataRequest>(request, std::move(sendUsingMe));
          }));

  forMe.registerHandler(
      StoDispatchKeys_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoDispatchKeys>>(
          [&](const Handle<pdb::StoDispatchKeys>& request, PDBCommunicatorPtr sendUsingMe) {
            return handleDispatchedData<PDBCommunicator, RequestFactory, pdb::StoDispatchKeys, StoStoreKeysRequest>(request, std::move(sendUsingMe));
          }));

  forMe.registerHandler(
      StoGetSetPagesRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoGetSetPagesRequest>>([&](const pdb::Handle<pdb::StoGetSetPagesRequest>& request, PDBCommunicatorPtr sendUsingMe) {
        return handleGetSetPages<PDBCommunicator, RequestFactory>(request, std::move(sendUsingMe));
      }));

  forMe.registerHandler(
      StoMaterializeKeysRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoMaterializeKeysRequest>>([&](const pdb::Handle<pdb::StoMaterializeKeysRequest>& request, PDBCommunicatorPtr sendUsingMe) {
        return handleMaterializeKeysSet<PDBCommunicator, RequestFactory>(request, std::move(sendUsingMe));
      }));

  forMe.registerHandler(
      StoMaterializePageSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoMaterializePageSetRequest>>([&](const pdb::Handle<pdb::StoMaterializePageSetRequest>& request, PDBCommunicatorPtr sendUsingMe) {
        return handleMaterializeSet<PDBCommunicator, RequestFactory>(request, std::move(sendUsingMe));
      }));

  forMe.registerHandler(
      StoRemovePageSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoRemovePageSetRequest>>([&](Handle<pdb::StoRemovePageSetRequest> request, PDBCommunicatorPtr sendUsingMe) {
        return handleRemovePageSet(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoStartFeedingPageSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoStartFeedingPageSetRequest>>([&](Handle<pdb::StoStartFeedingPageSetRequest> request, PDBCommunicatorPtr sendUsingMe) {
        return handleStartFeedingPageSetRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoClearSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoClearSetRequest>>([&](Handle<pdb::StoClearSetRequest> request, PDBCommunicatorPtr sendUsingMe) {
        return handleClearSetRequest(request, sendUsingMe);
  }));

  forMe.registerHandler(
      StoFetchSetPagesRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoFetchSetPagesRequest>>([&](Handle<pdb::StoFetchSetPagesRequest> request, PDBCommunicatorPtr sendUsingMe) {
        return handleStoFetchSetPages(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoFetchPageSetPagesRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoFetchPageSetPagesRequest>>([&](Handle<pdb::StoFetchPageSetPagesRequest> request, PDBCommunicatorPtr sendUsingMe) {
        return handleStoFetchPageSetPagesRequest(request, sendUsingMe);
      }));

  // backend
  forMe.registerHandler(
      StoStoreDataRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoStoreDataRequest>>(
          [&](const Handle<pdb::StoStoreDataRequest>& request, PDBCommunicatorPtr sendUsingMe) {
            return handleStoreData(request, sendUsingMe);
          }));

  forMe.registerHandler(
      StoStoreKeysRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoStoreKeysRequest>>(
          [&](Handle<pdb::StoStoreKeysRequest> request, PDBCommunicatorPtr sendUsingMe) {
            return handleStoreKeys(request, sendUsingMe);
          }));

  forMe.registerHandler(
      StoRemovePageSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoRemovePageSetRequest>>(
          [&](Handle<pdb::StoRemovePageSetRequest> request, PDBCommunicatorPtr sendUsingMe) {
            return handlePageSet(request, sendUsingMe);
          }));

  forMe.registerHandler(
      StoStartFeedingPageSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoStartFeedingPageSetRequest>>(
          [&](Handle<pdb::StoStartFeedingPageSetRequest> request, PDBCommunicatorPtr sendUsingMe) {
            return handleStartFeedingPageSetRequest(request, sendUsingMe);
          }));
}

std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleStoFetchSetPages(pdb::Handle<pdb::StoFetchSetPagesRequest> &request,
                                                                                    std::shared_ptr<PDBCommunicator> &sendUsingMe) {

  /// 1. Get all the pages we need from the set

  // make the set identifier
  PDBSetPtr storageSet = nullptr;
  if(!request->forKeys) {

    // we are requesting a regular set
    storageSet = std::make_shared<PDBSet>(request->databaseName, request->setName);
  }
  else {

    // we request the key modify the set name
    storageSet = std::make_shared<PDBSet>(request->databaseName, PDBCatalog::toKeySetName(request->setName));
  }

  // get the pages of the set
  std::vector<uint64_t> pages;
  {
    // lock the stuff
    unique_lock<std::mutex> lck(pageMutex);

    // try to find the page
    auto it = this->pageStats.find(storageSet);
    if(it != this->pageStats.end()) {

      // reserve the pages
      pages.reserve(it->second.lastPage);

      // do we even have this page
      uint64_t currPage = 0;
      while(currPage <= it->second.lastPage) {

        // check if the page is valid
        if(pageExists(storageSet, currPage) && !isPageBeingWrittenTo(storageSet, currPage) && !isPageFree(storageSet, currPage)) {
          pages.emplace_back(currPage);
        }

        // if not try to go to the next one
        currPage++;
      }
    }
  }

  /// 2. Send a response with the number of pages to expect

  // make the allocation
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // make the response
  pdb::Handle<StoFetchPagesResponse> response = pdb::makeObject<StoFetchPagesResponse>(pages.size());

  // send it
  std::string error;
  sendUsingMe->sendObject(response, error);

  // get the buffer manager
  auto bufferManager = getFunctionalityPtr<PDBBufferManagerInterface>();

  // make the response
  pdb::Handle<StoFetchNextPageResult> fetchNextPage = pdb::makeObject<StoFetchNextPageResult>();

  // go through each page
  for(const auto &page : pages) {

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
    std::cout << "Sending\n";
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

std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleStoFetchPageSetPagesRequest(pdb::Handle<pdb::StoFetchPageSetPagesRequest> &request,
                                                                                               std::shared_ptr<PDBCommunicator> &sendUsingMe){

  throw runtime_error("Fetching pages from a page set is not yet supported");
}

bool pdb::PDBStorageManagerFrontend::isPageBeingWrittenTo(const pdb::PDBSetPtr &set, uint64_t pageNum) {

  // try to find it
  auto it = pagesBeingWrittenTo.find(set);

  // do we even have it here
  if(it == pagesBeingWrittenTo.end()) {
    return false;
  }

  // check if is in the set of free pages
  return it->second.find(pageNum) != it->second.end();}

bool pdb::PDBStorageManagerFrontend::isPageFree(const pdb::PDBSetPtr &set, uint64_t pageNum) {

  // try to find it
  auto it = freeSkippedPages.find(set);

  // do we even have it here
  if(it == freeSkippedPages.end()) {
    return false;
  }

  // check if is in the set of free pages
  return it->second.find(pageNum) != it->second.end();
}

bool pdb::PDBStorageManagerFrontend::pageExists(const pdb::PDBSetPtr &set, uint64_t pageNum) {

  // try to find the page
  auto it = this->pageStats.find(set);

  // if it exists and is smaller or equal to the last page then it exists
  return it != this->pageStats.end() && pageNum <= it->second.lastPage;
}

std::pair<bool, uint64_t> pdb::PDBStorageManagerFrontend::getValidPage(const pdb::PDBSetPtr &set, uint64_t pageNum) {

  // lock the stuff
  unique_lock<std::mutex> lck(pageMutex);

  // try to find the page
  auto it = this->pageStats.find(set);

  // do we even have stats about this, if not finish
  if(it == this->pageStats.end()) {
    return make_pair(false, 0);
  }

  // do we even have this page
  while(pageNum <= it->second.lastPage) {

    // check if the page is valid
    if(pageExists(set, pageNum) && !isPageBeingWrittenTo(set, pageNum) && !isPageFree(set, pageNum)) {
      return make_pair(true, pageNum);
    }

    // if not try to go to the next one
    pageNum++;
  }

  // finish
  return make_pair(false, 0);
}

uint64_t pdb::PDBStorageManagerFrontend::getNextFreePage(const pdb::PDBSetPtr &set) {

  // see if we have a free page already
  auto pages = freeSkippedPages.find(set);
  if(!pages->second.empty()) {

    // get the page number
    auto page = *pages->second.begin();

    // remove the thing
    pages->second.erase(pages->second.begin());

    // return the page
    return page;
  }

  // try to find the set
  auto it = pageStats.find(set);

  // do we even have a record for this set
  uint64_t pageNum;
  if(it == pageStats.end()) {

    // set the page to zero since this is the first page
    pageStats[set].lastPage = 0;
    pageNum = 0;

    // set the page size
    pageStats[set].size = 0;
  }
  else {

    // increment the last page
    pageNum = ++it->second.lastPage;
  }

  return pageNum;
}

void pdb::PDBStorageManagerFrontend::incrementSetSize(const pdb::PDBSetPtr &set, uint64_t uncompressedSize) {

  // try to find the set
  auto it = pageStats.find(set);

  // increment the set size on this node
  it->second.size += uncompressedSize;
}

void pdb::PDBStorageManagerFrontend::freeSetPage(const pdb::PDBSetPtr &set, uint64_t pageNum) {

  // insert the page into the free list
  freeSkippedPages[set].insert(pageNum);
}

void pdb::PDBStorageManagerFrontend::startWritingToPage(const pdb::PDBSetPtr &set, uint64_t pageNum) {
  // mark the page as being written to
  pagesBeingWrittenTo[set].insert(pageNum);
}

void pdb::PDBStorageManagerFrontend::endWritingToPage(const pdb::PDBSetPtr &set, uint64_t pageNum) {
  // unmark the page as being written to
  pagesBeingWrittenTo[set].erase(pageNum);
}

bool pdb::PDBStorageManagerFrontend::handleDispatchFailure(const PDBSetPtr &set, uint64_t pageNum, uint64_t size, const PDBCommunicatorPtr& communicator) {

  // where we put the error
  std::string error;

  {
    // lock the stuff
    unique_lock<std::mutex> lck(pageMutex);

    // finish writing to the set
    endWritingToPage(set, pageNum);

    // return the page to the free list
    freeSetPage(set, pageNum);

    // decrement back the set size
    decrementSetSize(set, size);
  }

  // create an allocation block to hold the response
  Handle<SimpleRequestResult> failResponse = makeObject<SimpleRequestResult>(false, error);

  // sends result to requester
  return communicator->sendObject(failResponse, error);
}

void pdb::PDBStorageManagerFrontend::decrementSetSize(const pdb::PDBSetPtr &set, uint64_t uncompressedSize) {

  // try to find the set
  auto it = pageStats.find(set);

  // increment the set size on this node
  it->second.size -= uncompressedSize;
}

std::shared_ptr<pdb::PDBStorageSetStats> pdb::PDBStorageManagerFrontend::getSetStats(const PDBSetPtr &set) {

  // try to find the set
  auto it = pageStats.find(set);

  // if we have it return it
  if(it != pageStats.end()){
    return std::make_shared<pdb::PDBStorageSetStats>(it->second);
  }

  // return null
  return nullptr;
}


pdb::PDBSetPageSetPtr pdb::PDBStorageManagerFrontend::createPageSetFromPDBSet(const std::string &db, const std::string &set, bool isKeyed) {


  // get the configuration
  auto conf = this->getConfiguration();

  /// 1. Contact the frontend and to get the number of pages

  auto pageInfo = RequestFactory::heapRequest<StoGetSetPagesRequest, StoGetSetPagesResult, std::pair<bool, std::vector<uint64_t>>>(
      logger, conf->port, conf->address, std::make_pair<bool, std::vector<uint64_t>>(false, std::vector<uint64_t>()), 1024,
      [&](Handle<StoGetSetPagesResult> result) {

        // do we have a result if not return false
        if (result == nullptr) {

          logger->error("Failed to get the number of pages for a page set created for the following PDBSet : (" + db + "," + set + ")");
          return std::make_pair<bool, std::vector<uint64_t>>(false, std::vector<uint64_t>());
        }

        // did we succeed
        if (!result->success) {

          logger->error("Failed to get the number of pages for a page set created for the following PDBSet : (" + db + "," + set + ")");
          return std::make_pair<bool, std::vector<uint64_t>>(false, std::vector<uint64_t>());
        }

        // copy the stuff
        std::vector<uint64_t> pages;
        pages.reserve(result->pages.size());
        for(int i = 0; i < result->pages.size(); ++i) { pages.emplace_back(result->pages[i]); }

        // we succeeded
        return std::make_pair(result->success, std::move(pages));
      }, db, set, isKeyed);

  // if we failed return a null ptr
  if(!pageInfo.first) {
    return nullptr;
  }

  /// 3. Create it and return it

  if(!isKeyed) {

    // store the page set
    return std::make_shared<pdb::PDBSetPageSet>(db, set, pageInfo.second, getFunctionalityPtr<PDBBufferManagerInterface>());
  }
  else {

    // store the page set
    return std::make_shared<pdb::PDBSetPageSet>(db, PDBCatalog::toKeySetName(set), pageInfo.second, getFunctionalityPtr<PDBBufferManagerInterface>());
  }
}

pdb::PDBAnonymousPageSetPtr pdb::PDBStorageManagerFrontend::createAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID) {

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

pdb::PDBRandomAccessPageSetPtr pdb::PDBStorageManagerFrontend::createRandomAccessPageSet(const std::pair<uint64_t, std::string> &pageSetID) {

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

pdb::PDBFeedingPageSetPtr pdb::PDBStorageManagerFrontend::createFeedingAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID, uint64_t numReaders, uint64_t numFeeders) {

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

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManagerFrontend::fetchPDBSet(const std::string &database,
                                                                      const std::string &set,
                                                                      bool isKey,
                                                                      const std::string &ip,
                                                                      int32_t port) {


  // get the configuration
  auto conf = this->getConfiguration();

  /// 1. Contact the frontend to establish a connection

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // the communicator
  PDBCommunicatorPtr comm = make_shared<PDBCommunicator>();
  string errMsg;

  // connect to the node
  if (!comm->connectToInternetServer(logger, port, ip, errMsg)) {

    // log the error
    logger->error(errMsg);
    logger->error("Could not connect node " + ip + ":" + std::to_string(port) + " to fetch the set (" +  database + ":" + set + ").\n");

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

  auto storageManager = getFunctionalityPtr<PDBStorageManagerFrontend>();
  auto bufferManager = getFunctionalityPtr<PDBBufferManagerInterface>();

  // return the fetching page set
  return std::make_shared<pdb::PDBFetchingPageSet>(comm,
                                                   storageManager,
                                                   bufferManager,
                                                   numPages);
}

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManagerFrontend::fetchPageSet(const PDBSourcePageSetSpec &pageSetSpec,
                                                                       bool isKey,
                                                                       const std::string &ip,
                                                                       int32_t port) {
  // get the configuration
  auto conf = this->getConfiguration();

  /// 1. Contact the frontend to establish a connection

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // the communicator
  PDBCommunicatorPtr comm = make_shared<PDBCommunicator>();
  string errMsg;

  // connect to the node
  if (!comm->connectToInternetServer(logger, port, ip, errMsg)) {

    // log the error
    logger->error(errMsg);
    logger->error("Could not connect node " + ip + ":" + std::to_string(port) +
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
                                                   getFunctionalityPtr<PDBStorageManagerFrontend>(),
                                                   getFunctionalityPtr<PDBBufferManagerInterface>(),
                                                   numPages);
}

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManagerFrontend::getPageSet(const std::pair<uint64_t, std::string> &pageSetID) {

  // try to find the page if it exists return it
  auto it = pageSets.find(pageSetID);
  if(it != pageSets.end()) {
    return std::dynamic_pointer_cast<PDBAbstractPageSet>(it->second);
  }

  // return null since we don't have it
  return nullptr;
}

bool pdb::PDBStorageManagerFrontend::removePageSet(const std::pair<uint64_t, std::string> &pageSetID) {

  // erase it if it exists
  return pageSets.erase(pageSetID) == 1;
}

bool pdb::PDBStorageManagerFrontend::materializePageSet(const pdb::PDBAbstractPageSetPtr& pageSet, const std::pair<std::string, std::string> &set) {

  // if the page set is empty no need materialize stuff
  if(pageSet->getNumPages() == 0) {
    return true;
  }

  // result indicators
  std::string error;
  bool success = true;

  /// 1. Connect to the frontend

  // the communicator
  std::shared_ptr<PDBCommunicator> comm = std::make_shared<PDBCommunicator>();

  // try to connect
  int numRetries = 0;
  while (!comm->connectToInternetServer(logger, getConfiguration()->port, getConfiguration()->address, error)) {

    // are we out of retires
    if(numRetries > getConfiguration()->maxRetries) {

      // log the error
      logger->error(error);
      logger->error("Can not connect to remote server with port=" + std::to_string(getConfiguration()->port) + " and address="+ getConfiguration()->address + ");");

      // set the success
      success = false;
      break;
    }

    // increment the number of retries
    numRetries++;
  }

  // if we failed
  if(!success) {

    // log the error
    logger->error("We failed to to connect to the frontend in order to materialize the page set.");

    // ok this sucks return false
    return false;
  }

  /// 2. Make a request to materialize page set

  // make an allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{1024};

  // set the stat results
  pdb::Handle<StoMaterializePageSetRequest> materializeRequest = pdb::makeObject<StoMaterializePageSetRequest>(set.first,
                                                                                                               set.second,
                                                                                                               pageSet->getNumRecords(),
                                                                                                               pageSet->getNumPages());

  // sends result to requester
  success = comm->sendObject(materializeRequest, error);

  // check if we failed
  if(!success) {

    // log the error
    logger->error(error);

    // ok this sucks we are out of here
    return false;
  }

  /// 3. Wait for an ACK

  // wait for the storage finish result
  std::vector<int64_t> pages;
  success = RequestFactory::waitHeapRequest<StoMaterializePageSetResult, bool>(logger, comm, false,
                                                                               [&](const Handle<StoMaterializePageSetResult>& result) {

                                                                                 // check the result
                                                                                 if (result != nullptr) {

                                                                                   // move the pages
                                                                                   pages.resize(result->pageNumbers.size());
                                                                                   for(int32_t i = 0; i < result->pageNumbers.size(); ++i) {
                                                                                     pages[i] = result->pageNumbers[i];
                                                                                   }

                                                                                   // finish
                                                                                   return true;
                                                                                 }

                                                                                 // set the error
                                                                                 error = "Failed to get the free pages\n";

                                                                                 // we failed return so
                                                                                 return false;
                                                                               });

  // check if we failed
  if(!success) {

    // log the error
    logger->error(error);

    // ok this sucks we are out of here
    return false;
  }

  /// 4. Grab the pages from the frontend

  // buffer manager
  auto bufferManager = getFunctionalityPtr<pdb::PDBBufferManagerInterface>();
  auto setIdentifier = std::make_shared<PDBSet>(set.first, set.second);

  // go through each page and materialize
  PDBPageHandle page;
  auto numPages = pageSet->getNumPages();
  uint64_t size = 0;
  for (int i = 0; i < numPages; ++i) {

    /// 4.1 Grab a page from the page set

    // grab the next page
    page = pageSet->getNextPage(0);

    // increment the size
    size += page->getSize();

    /// 4.2 Move the page to set
    bufferManager->moveAnonymousPagesToSet(setIdentifier, pages[i], page);
  }

  /// 5. Now update the set statistics

  // broadcast the set size change so far
  PDBCatalogClient pdbClient(getConfiguration()->managerPort, getConfiguration()->managerAddress, logger);
  success = pdbClient.incrementSetRecordInfo(getConfiguration()->getNodeIdentifier(), set.first, set.second, size, pageSet->getNumRecords(), error);

  /// 6. Finish this, by sending an ack

  // create an allocation block to hold the response
  Handle<SimpleRequestResult> simpleResponse = makeObject<SimpleRequestResult>(success, error);

  // set the response
  simpleResponse->res = success;
  simpleResponse->errMsg = error;

  // sends result to requester
  comm->sendObject(simpleResponse, error);

  // we succeeded
  return success;
}

bool pdb::PDBStorageManagerFrontend::materializeKeys(const pdb::PDBAbstractPageSetPtr &pageSet,
                                                    const std::pair<std::string, std::string> &mainSet,
                                                    const pdb::PDBKeyExtractorPtr &keyExtractor) {

//  // if the page set is empty no need materialize stuff
//  if(pageSet->getNumPages() == 0) {
//    return true;
//  }
//
//  //
//  auto dbName = mainSet.first;
//  auto setName = PDBCatalog::toKeySetName(mainSet.second);
//
//  // result indicators
//  std::string error;
//  bool success = true;
//
//  /// 1. Connect to the frontend
//
//  // the communicator
//  std::shared_ptr<PDBCommunicator> comm = std::make_shared<PDBCommunicator>();
//
//  // try to connect
//  int numRetries = 0;
//  while (!comm->connectToInternetServer(logger, getConfiguration()->port, getConfiguration()->address, error)) {
//
//    // are we out of retires
//    if(numRetries > getConfiguration()->maxRetries) {
//
//      // log the error
//      logger->error(error);
//      logger->error("Can not connect to remote server with port=" + std::to_string(getConfiguration()->port) + " and address="+ getConfiguration()->address + ");");
//
//      // set the success
//      success = false;
//      break;
//    }
//
//    // increment the number of retries
//    numRetries++;
//  }
//
//  // if we failed
//  if(!success) {
//
//    // log the error
//    logger->error("We failed to to connect to the frontend in order to materialize the page set.");
//
//    // ok this sucks return false
//    return false;
//  }
//
//  /// 2. Make a request to materialize page set
//
//  // make an allocation block
//  const pdb::UseTemporaryAllocationBlock tempBlock{1024};
//
//  // set the stat results
//  pdb::Handle<StoMaterializeKeysRequest> materializeRequest = pdb::makeObject<StoMaterializeKeysRequest>(dbName, setName, pageSet->getNumRecords());
//
//  // sends result to requester
//  success = comm->sendObject(materializeRequest, error);
//
//  // check if we failed
//  if(!success) {
//
//    // log the error
//    logger->error(error);
//
//    // ok this sucks we are out of here
//    return false;
//  }
//
//  /// 3. Wait for an ACK
//
//  // wait for the storage finish result
//  success = RequestFactory::waitHeapRequest<SimpleRequestResult, bool>(logger, comm, false,
//                                                                       [&](const Handle<SimpleRequestResult>& result) {
//
//                                                                         // check the result
//                                                                         if (result != nullptr && result->getRes().first) {
//
//                                                                           // finish
//                                                                           return true;
//                                                                         }
//
//                                                                         // set the error
//                                                                         error = result->getRes().second;
//
//                                                                         // we failed return so
//                                                                         return false;
//                                                                       });
//
//  // check if we failed
//  if(!success) {
//
//    // log the error
//    logger->error(error);
//
//    // ok this sucks we are out of here
//    return false;
//  }
//
//  /// 4. Grab the pages from the frontend
//
//  // buffer manager
//  auto bufferManager = getFunctionalityPtr<pdb::PDBBufferManagerInterface>();
//  auto setIdentifier = std::make_shared<PDBSet>(dbName, setName);
//
//  // go through each page and materialize
//  PDBPageHandle inputPage;
//  auto numPages = pageSet->getNumPages();
//
//  // while we still have pages and we have processed last one
//  while(numPages != 0 || !keyExtractor->processedLast()) {
//
//    // grab a page
//    auto keyPage = bufferManager->expectPage(comm);
//
//    // check if we got a page
//    if(keyPage == nullptr) {
//
//      // log it
//      logger->error("Failed to get the page from the frontend when materializing a set!");
//
//      // finish
//      return false;
//    }
//
//    // make the allocation block
//    makeObjectAllocatorBlock(keyPage->getBytes(), keyPage->getSize(), true);
//
//    // fetch the first page
//    inputPage = pageSet->getNextPage(0);
//
//    // process all the pages
//    while(numPages != 0) {
//
//      // did we processed the last?
//      if(keyExtractor->processedLast()) {
//
//        // grab the next page
//        inputPage = pageSet->getNextPage(0);
//      }
//
//      // grabbed the next page
//      numPages--;
//
//      // repin the page
//      inputPage->repin();
//
//      // copy the memory to the set page
//      try {
//
//        // extract the keys from the page
//        keyExtractor->extractKeys(inputPage, keyPage);
//      }
//      catch (pdb::NotEnoughSpace &n) {}
//
//      // unpin the page
//      inputPage->unpin();
//    }
//
//    // get the
//    int32_t pageSize = keyExtractor->pageSize(keyPage);
//
//    // make an allocation block to send the response
//    const pdb::UseTemporaryAllocationBlock blk{1024};
//
//    // make a request to mark that we succeeded
//    pdb::Handle<StoMaterializePageResult> materializeResult = pdb::makeObject<StoMaterializePageResult>(dbName, setName, pageSize, true, (numPages != 0 || !keyExtractor->processedLast()) );
//
//    // sends result to requester
//    success = comm->sendObject(materializeResult, error);
//
//    // did the request succeed if so we are done
//    if(!success) {
//
//      // log it
//      logger->error(error);
//
//      // finish here
//      return false;
//    }
//  }
//
//  /// 5. Wait for an ACK
//
//  // wait for the storage finish result
//  success = RequestFactory::waitHeapRequest<SimpleRequestResult, bool>(logger, comm, false,[&](const Handle<SimpleRequestResult>& result) {
//
//    // check the result
//    if (result != nullptr && result->getRes().first) {
//
//      // finish
//      return true;
//    }
//
//    // set the error
//    error = result->getRes().second;
//
//    // we failed return so
//    return false;
//  });
//
//  // check if we failed
//  if(!success) {
//
//    // log the error
//    logger->error(error);
//
//    // ok this sucks we are out of here
//    return false;
//  }

  // we succeeded
  return true;
}