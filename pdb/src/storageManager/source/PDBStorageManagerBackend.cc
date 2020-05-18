//
// Created by dimitrije on 2/11/19.
//

#include <SharedEmployee.h>
#include <memory>
#include "HeapRequestHandler.h"
#include "StoStoreDataRequest.h"
#include "StoGetSetPagesRequest.h"
#include "StoGetSetPagesResult.h"
#include <boost/filesystem/path.hpp>
#include <PDBSetPageSet.h>
#include <PDBStorageManagerBackend.h>
#include <StoMaterializePageSetRequest.h>
#include <StoRemovePageSetRequest.h>
#include <StoMaterializePageResult.h>
#include <PDBBufferManagerBackEnd.h>
#include <StoStartFeedingPageSetRequest.h>
#include <StoStoreKeysRequest.h>
#include <StoFetchSetPagesRequest.h>
#include <PDBFetchingPageSet.h>
#include <StoFetchPageSetPagesRequest.h>
#include <StoFetchPagesResponse.h>
#include <PDBLabeledPageSet.h>
#include <StoMaterializeKeysRequest.h>

void pdb::PDBStorageManagerBackend::init() {

  // init the class
  logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "logs").string(), "PDBStorageManagerBackend.log");
}

void pdb::PDBStorageManagerBackend::registerHandlers(PDBServer &forMe) {

  forMe.registerHandler(
      StoStoreDataRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoStoreDataRequest>>(
          [&](Handle<pdb::StoStoreDataRequest> request, PDBCommunicatorPtr sendUsingMe) {
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

pdb::PDBSetPageSetPtr pdb::PDBStorageManagerBackend::createPageSetFromPDBSet(const std::string &db, const std::string &set, bool isKeyed) {


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

pdb::PDBAnonymousPageSetPtr pdb::PDBStorageManagerBackend::createAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID) {

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

pdb::PDBRandomAccessPageSetPtr pdb::PDBStorageManagerBackend::createRandomAccessPageSet(const std::pair<uint64_t, std::string> &pageSetID) {

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

pdb::PDBFeedingPageSetPtr pdb::PDBStorageManagerBackend::createFeedingAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID, uint64_t numReaders, uint64_t numFeeders) {

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

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManagerBackend::fetchPDBSet(const std::string &database,
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

  auto storageManager = getFunctionalityPtr<PDBStorageManagerBackend>();
  auto bufferManager = getFunctionalityPtr<PDBBufferManagerInterface>();

  // return the fetching page set
  return std::make_shared<pdb::PDBFetchingPageSet>(comm,
                                                   storageManager,
                                                   bufferManager,
                                                   numPages);
}

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManagerBackend::fetchPageSet(const PDBSourcePageSetSpec &pageSetSpec,
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
                                                   getFunctionalityPtr<PDBStorageManagerBackend>(),
                                                   getFunctionalityPtr<PDBBufferManagerInterface>(),
                                                   numPages);
}

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManagerBackend::getPageSet(const std::pair<uint64_t, std::string> &pageSetID) {

  // try to find the page if it exists return it
  auto it = pageSets.find(pageSetID);
  if(it != pageSets.end()) {
    return std::dynamic_pointer_cast<PDBAbstractPageSet>(it->second);
  }

  // return null since we don't have it
  return nullptr;
}

bool pdb::PDBStorageManagerBackend::removePageSet(const std::pair<uint64_t, std::string> &pageSetID) {

  // erase it if it exists
  return pageSets.erase(pageSetID) == 1;
}

bool pdb::PDBStorageManagerBackend::materializePageSet(const pdb::PDBAbstractPageSetPtr& pageSet, const std::pair<std::string, std::string> &set) {

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
  pdb::Handle<StoMaterializePageSetRequest> materializeRequest = pdb::makeObject<StoMaterializePageSetRequest>(set.first, set.second, pageSet->getNumRecords());

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
  success = RequestFactory::waitHeapRequest<SimpleRequestResult, bool>(logger, comm, false,
    [&](const Handle<SimpleRequestResult>& result) {

      // check the result
      if (result != nullptr && result->getRes().first) {

        // finish
        return true;
      }

      // set the error
      error = result->getRes().second;

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
  pdb::PDBBufferManagerBackEndPtr bufferManager = std::dynamic_pointer_cast<PDBBufferManagerBackEndImpl>(getFunctionalityPtr<pdb::PDBBufferManagerInterface>());
  auto setIdentifier = std::make_shared<PDBSet>(set.first, set.second);

  // go through each page and materialize
  PDBPageHandle page;
  auto numPages = pageSet->getNumPages();
  for (int i = 0; i < numPages; ++i) {

    // grab the next page
    page = pageSet->getNextPage(0);

    // repin the page
    page->repin();

    // grab a page
    auto setPage = bufferManager->expectPage(comm);

    // check if we got a page
    if(setPage == nullptr) {

      // log it
      logger->error("Failed to get the page from the frontend when materializing a set!");

      // finish
      return false;
    }

    // get the size of the page
    auto pageSize = page->getSize();

    // copy the memory to the set page
    memcpy(setPage->getBytes(), page->getBytes(), pageSize);

    // unpin the page
    page->unpin();

    // make an allocation block to send the response
    const pdb::UseTemporaryAllocationBlock blk{1024};

    // make a request to mark that we succeeded
    pdb::Handle<StoMaterializePageResult> materializeResult = pdb::makeObject<StoMaterializePageResult>(set.first, set.second, pageSize, true, (i + 1) < numPages);

    // sends result to requester
    success = comm->sendObject(materializeResult, error);

    // did the request succeed if so we are done
    if(!success) {

      // log it
      logger->error(error);

      // finish here
      return false;
    }
  }

  /// 5. Wait for an ACK

  // wait for the storage finish result
  success = RequestFactory::waitHeapRequest<SimpleRequestResult, bool>(logger, comm, false,[&](const Handle<SimpleRequestResult>& result) {

    // check the result
    if (result != nullptr && result->getRes().first) {

     // finish
     return true;
    }

    // set the error
    error = result->getRes().second;

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

  // we succeeded
  return true;
}

bool pdb::PDBStorageManagerBackend::materializeKeys(const pdb::PDBAbstractPageSetPtr &pageSet,
                                                    const std::pair<std::string, std::string> &mainSet,
                                                    const pdb::PDBKeyExtractorPtr &keyExtractor) {

  // if the page set is empty no need materialize stuff
  if(pageSet->getNumPages() == 0) {
    return true;
  }

  //
  auto dbName = mainSet.first;
  auto setName = PDBCatalog::toKeySetName(mainSet.second);

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
  pdb::Handle<StoMaterializeKeysRequest> materializeRequest = pdb::makeObject<StoMaterializeKeysRequest>(dbName, setName, pageSet->getNumRecords());

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
  success = RequestFactory::waitHeapRequest<SimpleRequestResult, bool>(logger, comm, false,
                                                                       [&](const Handle<SimpleRequestResult>& result) {

    // check the result
    if (result != nullptr && result->getRes().first) {

     // finish
     return true;
    }

    // set the error
    error = result->getRes().second;

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
  pdb::PDBBufferManagerBackEndPtr bufferManager = std::dynamic_pointer_cast<PDBBufferManagerBackEndImpl>(getFunctionalityPtr<pdb::PDBBufferManagerInterface>());
  auto setIdentifier = std::make_shared<PDBSet>(dbName, setName);

  // go through each page and materialize
  PDBPageHandle inputPage;
  auto numPages = pageSet->getNumPages();

  // while we still have pages and we have processed last one
  while(numPages != 0 || !keyExtractor->processedLast()) {

    // grab a page
    auto keyPage = bufferManager->expectPage(comm);

    // check if we got a page
    if(keyPage == nullptr) {

      // log it
      logger->error("Failed to get the page from the frontend when materializing a set!");

      // finish
      return false;
    }

    // make the allocation block
    makeObjectAllocatorBlock(keyPage->getBytes(), keyPage->getSize(), true);

    // process all the pages
    while(numPages != 0) {

      // did we processed the last?
      if(keyExtractor->processedLast()) {

        // grab the next page
        inputPage = pageSet->getNextPage(0);
      }

      // grabbed the next page
      numPages--;

      // repin the page
      inputPage->repin();

      // copy the memory to the set page
      try {

        // extract the keys from the page
        keyExtractor->extractKeys(inputPage, keyPage);
      }
      catch (pdb::NotEnoughSpace &n) {}

      // unpin the page
      inputPage->unpin();
    }

    // get the
    int32_t pageSize = keyExtractor->pageSize(keyPage);

    // make an allocation block to send the response
    const pdb::UseTemporaryAllocationBlock blk{1024};

    // make a request to mark that we succeeded
    pdb::Handle<StoMaterializePageResult> materializeResult = pdb::makeObject<StoMaterializePageResult>(dbName, setName, pageSize, true, (numPages != 0 || !keyExtractor->processedLast()) );

    // sends result to requester
    success = comm->sendObject(materializeResult, error);

    // did the request succeed if so we are done
    if(!success) {

      // log it
      logger->error(error);

      // finish here
      return false;
    }
  }

  /// 5. Wait for an ACK

  // wait for the storage finish result
  success = RequestFactory::waitHeapRequest<SimpleRequestResult, bool>(logger, comm, false,[&](const Handle<SimpleRequestResult>& result) {

    // check the result
    if (result != nullptr && result->getRes().first) {

      // finish
      return true;
    }

    // set the error
    error = result->getRes().second;

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

  // we succeeded
  return true;
}