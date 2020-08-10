//
// Created by dimitrije on 2/17/19.
//

#ifndef PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_H
#define PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_H

#include <PDBStorageManagerFrontend.h>
#include <HeapRequestHandler.h>
#include <StoDispatchData.h>
#include <PDBBufferManagerInterface.h>
#include <StoStoreDataRequest.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <fstream>
#include <HeapRequest.h>
#include <StoGetNextPageRequest.h>
#include <StoGetNextPageResult.h>
#include "CatalogServer.h"
#include <StoGetPageRequest.h>
#include <StoGetPageResult.h>
#include <StoGetSetPagesResult.h>
#include <StoRemovePageSetRequest.h>
#include <StoMaterializePageResult.h>
#include <StoFeedPageRequest.h>
#include <StoMaterializeKeysRequest.h>
#include <StoMaterializePageSetResult.h>
#include <PDBBufferManagerImpl.h>

template <class T>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleGetPageRequest(const pdb::Handle<pdb::StoGetPageRequest> &request,
                                                                                  std::shared_ptr<T>  &sendUsingMe) {

  /// 1. Check if we have a page

  // create the set identifier
  auto set = make_shared<pdb::PDBSet>(request->databaseName, request->setName);

  // find the if this page is valid if not try to find another one...
  auto res = getValidPage(set, request->page);

  // set the result
  bool hasPage = res.first;
  uint64_t pageNum = res.second;

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
  auto page = this->getFunctionalityPtr<PDBBufferManagerInterface>()->getPage(set, pageNum);

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
  pdb::Handle<pdb::StoGetPageResult> response = pdb::makeObject<pdb::StoGetPageResult>(compressedSize, pageNum, true);

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

template <class Communicator, class Requests, class RequestType, class ForwardRequestType>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleDispatchedData(pdb::Handle<RequestType> request,
                                                                                  std::shared_ptr<Communicator> sendUsingMe) {

  /// 1. Get the page from the distributed storage

  // the error
  std::string error;

//  // grab the buffer manager
//  auto bufferManager = std::dynamic_pointer_cast<pdb::PDBBufferManagerFrontEnd>(getFunctionalityPtr<pdb::PDBBufferManagerInterface>());
//
//  // figure out how large the compressed payload is
//  size_t numBytes = sendUsingMe->getSizeOfNextObject();
//
//  // grab a page to write this
//  auto page = bufferManager->getPage(numBytes);
//
//  // grab the bytes
//  auto success = sendUsingMe->receiveBytes(page->getBytes(), error);
//
//  // did we fail
//  if(!success) {
//
//    // create an allocation block to hold the response
//    const UseTemporaryAllocationBlock tempBlock{1024};
//    Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, error);
//
//    // sends result to requester
//    sendUsingMe->sendObject(response, error);
//
//    return std::make_pair(false, error);
//  }
//
//  // figure out the size so we can increment it
//  // check the uncompressed size
//  size_t uncompressedSize = 0;
//  snappy::GetUncompressedLength((char*) page->getBytes(), numBytes, &uncompressedSize);
//
//  /// 2. Figure out the page we want to put this thing onto
//
//  uint64_t pageNum;
//  {
//    // lock the stuff that keeps track of the last page
//    unique_lock<std::mutex> lck(pageMutex);
//
//    // make the set
//    auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);
//
//    // get the next page
//    pageNum = getNextFreePage(set);
//
//    // indicate that we are writing to this page
//    startWritingToPage(set, pageNum);
//
//    // increment the set size
//    incrementSetSize(set, uncompressedSize);
//  }
//
//  /// 3. Initiate the storing on the backend
//
//  PDBCommunicatorPtr communicatorToBackend = make_shared<PDBCommunicator>();
//  if (!communicatorToBackend->connectToLocalServer(logger, getConfiguration()->ipcFile, error)) {
//    return std::make_pair(false, error);
//  }
//
//  // create an allocation block to hold the response
//  const UseTemporaryAllocationBlock tempBlock{1024};
//  Handle<ForwardRequestType> response = makeObject<ForwardRequestType>(request->databaseName, request->setName, pageNum, request->compressedSize);
//
//  // send the thing to the backend
//  if (!communicatorToBackend->sendObject(response, error)) {
//
//    // make the set
//    auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);
//
//    // set the indicators and send a NACK to the client since we failed
//    handleDispatchFailure(set, pageNum, uncompressedSize, sendUsingMe);
//
//    // finish
//    return std::make_pair(false, error);
//  }
//
//  /// 4. Forward the page to the backend
//
//  // forward the page
//  if(!bufferManager->forwardPage(page, communicatorToBackend, error)) {
//
//    // we could not forward the page
//    auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);
//
//    // set the indicators and send a NACK to the client since we failed
//    handleDispatchFailure(set, pageNum, uncompressedSize, sendUsingMe);
//
//    // finish
//    return std::make_pair(false, error);
//  }
//
//  /// 5. Wait for the backend to finish the stuff
//
//  success = Requests::template waitHeapRequest<SimpleRequestResult, bool>(logger, communicatorToBackend, false,
//  [&](Handle<SimpleRequestResult> result) {
//
//   // check the result
//   if (result != nullptr && result->getRes().first) {
//     return true;
//   }
//
//   // since we failed just invalidate the set
//   auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);
//   {
//    // lock the stuff
//    unique_lock<std::mutex> lck(pageMutex);
//
//    // finish writing to the set
//    endWritingToPage(set, pageNum);
//
//    // return the page to the free list
//    freeSetPage(set, pageNum);
//
//    // decrement the size of the set
//    decrementSetSize(set, uncompressedSize);
//   }
//
//   // log the error
//   error = "Error response from distributed-storage: " + result->getRes().second;
//   logger->error("Error response from distributed-storage: " + result->getRes().second);
//
//   return false;
//  });
//
//  // finish writing to the set
//  endWritingToPage(std::make_shared<PDBSet>(request->databaseName, request->setName), pageNum);
//
//  /// 6. Send the response that we are done
//
//  // create an allocation block to hold the response
//  Handle<SimpleRequestResult> simpleResponse = makeObject<SimpleRequestResult>(success, error);
//
//  // sends result to requester
//  success = sendUsingMe->sendObject(simpleResponse, error) && success;

  // finish
  bool success = false;
  return std::make_pair(success, error);
}

template<class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleGetSetPages(pdb::Handle<pdb::StoGetSetPagesRequest> request, shared_ptr<Communicator> sendUsingMe) {

  // the error
  std::string error;
  bool success = true;

  // check if the set exists
  auto set = getFunctionalityPtr<pdb::PDBCatalogClient>()->getSet(request->databaseName, request->setName, error);
  if(set == nullptr) {

    // set the error
    error = "The set the pages were requested does not exist!";
    success = false;
  }

  // check if we requested a keyed set but the set is not keyed
  if(success && request->forKeys && !set->isStoringKeys) {

    // set the error
    error = "Requested the keys of a set but the keys don't exist!";
    success = false;
  }

  // check if everything went well
  if(!success) {

    // make an allocation block
    const UseTemporaryAllocationBlock tempBlock{1024};

    // make a NACK response
    pdb::Handle<pdb::StoGetSetPagesResult> result = pdb::makeObject<pdb::StoGetSetPagesResult>();

    // sends result to requester
    sendUsingMe->sendObject(result, error);

    // return the result
    return std::make_pair(success, error);
  }

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

  // make an allocation block
  const UseTemporaryAllocationBlock tempBlock{pages.size() * sizeof(uint64_t) + 1024};

  // make the result
  pdb::Handle<pdb::StoGetSetPagesResult> result = pdb::makeObject<pdb::StoGetSetPagesResult>(pages, true);

  // sends result to requester
  success = sendUsingMe->sendObject(result, error);

  // return the result
  return std::make_pair(success, error);
}

template<class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleMaterializeSet(const pdb::Handle<pdb::StoMaterializePageSetRequest>& request,
                                                                                  shared_ptr<Communicator> sendUsingMe) {

  /// TODO this has to be more robust, right now this is just here to do the job!
  // success indicators
  bool success = true;
  std::string error;

  /// 1. Check if the set exists

  // check if the set exists
  if(!getFunctionalityPtr<pdb::PDBCatalogClient>()->setExists(request->databaseName, request->setName)) {

    // set the error
    error = "The set requested to materialize results does not exist!";
    success = false;
  }

  /// 2. Send all the page numbers that we need to materialize to

  // make an allocation block
  const UseTemporaryAllocationBlock tempBlock{1024};

  // make a result
  pdb::Handle<StoMaterializePageSetResult> materializeResult = pdb::makeObject<StoMaterializePageSetResult>(request->numPages);

  // make the set
  auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

  {
    // lock to do the bookkeeping and get all pages
    unique_lock<std::mutex> lck(pageMutex);
    for(int32_t i = 0; i < request->numPages; ++i) {

      // get the next page
      uint64_t pageNum = getNextFreePage(set);

      // store the page
      materializeResult->pageNumbers[i] = pageNum;

      // indicate that we are writing to this page
      startWritingToPage(set, pageNum);
    }
  }

  // sends result to requester
  success = sendUsingMe->sendObject(materializeResult, error);

  /// 3. Wait for an ACK or NACK that everything went fine

  if(success) {

    // wait for the storage finish result
    success = RequestFactory::waitHeapRequest<SimpleRequestResult, bool>(logger, sendUsingMe, false, [&](const Handle<SimpleRequestResult>& result) {

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
  }

  /// 4. End writting to these pages

  // lock to do the bookkeeping
  {
    // lock to do the bookkeeping and get all pages
    unique_lock<std::mutex> lck(pageMutex);
    for(int32_t i = 0; i < request->numPages; ++i) {

      // end writing to this page
      endWritingToPage(set, materializeResult->pageNumbers[i]);
    }
  }

  // return the result
  return std::make_pair(success, error);
}

template<class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleMaterializeKeysSet(const pdb::Handle<pdb::StoMaterializeKeysRequest>& request,
                                                                                      shared_ptr<Communicator> sendUsingMe) {

  /// TODO this has to be more robust, right now this is just here to do the job!
  // success indicators
  bool success = true;
  std::string error;
//
//  /// 1. Check if the set exists
//
//  // check if the set exists
//  if(!getFunctionalityPtr<pdb::PDBCatalogClient>()->setExists(request->databaseName, PDBCatalog::fromKeySetNameToSetName(request->setName))) {
//
//    // set the error
//    error = "The set requested to materialize results does not exist!";
//    success = false;
//  }
//
//  /// 2. send an ACK or NACK depending on whether the set exists
//
//  // make an allocation block
//  const UseTemporaryAllocationBlock tempBlock{1024};
//
//  // create an allocation block to hold the response
//  Handle<SimpleRequestResult> simpleResponse = makeObject<SimpleRequestResult>(success, error);
//
//  // sends result to requester
//  sendUsingMe->sendObject(simpleResponse, error);
//
//  // if we failed end here
//  if(!success) {
//
//    // return the result
//    return std::make_pair(success, error);
//  }
//
//  /// 3. Send pages over the wire to the backend
//
//  // grab the buffer manager
//  auto bufferManager = std::dynamic_pointer_cast<pdb::PDBBufferManagerFrontEnd>(getFunctionalityPtr<pdb::PDBBufferManagerInterface>());
//
//  // make the set
//  auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);
//
//  // this is going to count the total size of the pages
//  uint64_t totalSize = 0;
//
//  // this is the count of total records in this page set
//  uint64_t totalRecords = request->numRecords;
//
//  // start forwarding the pages
//  bool hasNext = true;
//  while (hasNext) {
//
//    uint64_t pageNum;
//    {
//      // lock to do the bookkeeping
//      unique_lock<std::mutex> lck(pageMutex);
//
//      // get the next page
//      pageNum = getNextFreePage(set);
//
//      // indicate that we are writing to this page
//      startWritingToPage(set, pageNum);
//    }
//
//    // get the page
//    auto page = bufferManager->getPage(set, pageNum);
//
//    // forward the page to the backend
//    success = bufferManager->forwardPage(page, sendUsingMe, error);
//
//    // did we fail?
//    if(!success) {
//
//      // lock to do the bookkeeping
//      unique_lock<std::mutex> lck(pageMutex);
//
//      // finish writing to the set
//      endWritingToPage(set, pageNum);
//
//      // return the page to the free list
//      freeSetPage(set, pageNum);
//
//      // free the lock
//      lck.unlock();
//
//      // do we need to update the set size
//      if(totalSize != 0) {
//
//        // broadcast the set size change so far
//        PDBCatalogClient pdbClient(getConfiguration()->managerPort, getConfiguration()->managerAddress, logger);
//        pdbClient.incrementKeyRecordInfo(getConfiguration()->getNodeIdentifier(),
//                                         request->databaseName,
//                                         PDBCatalog::fromKeySetNameToSetName(std::string(request->setName)),
//                                         totalSize,
//                                         totalRecords,
//                                         error);
//      }
//
//      // finish here since this is not recoverable on the backend
//      return std::make_pair(success, "Error occurred while forwarding the page to the backend.\n" + error);
//    }
//
//    // the size we want to freeze this thing to
//    size_t freezeSize = 0;
//
//    // wait for the storage finish result
//    success = RequestFactory::waitHeapRequest<StoMaterializePageResult, bool>(logger, sendUsingMe, false,
//      [&](const Handle<StoMaterializePageResult>& result) {
//
//        // check the result
//        if (result != nullptr && result->success) {
//
//          // set the freeze size
//          freezeSize = result->materializeSize;
//
//          // set the has next
//          hasNext = result->hasNext;
//
//          // finish
//          return result->success;
//        }
//
//        // set the error
//        error = "Backend materializing the page failed!";
//
//        // we failed return so
//        return false;
//    });
//
//    // did we fail?
//    if(!success) {
//
//      // lock to do the bookkeeping
//      unique_lock<std::mutex> lck(pageMutex);
//
//      // finish writing to the set
//      endWritingToPage(set, pageNum);
//
//      // return the page to the free list
//      freeSetPage(set, pageNum);
//
//      // free the lock
//      lck.unlock();
//
//      // do we need to update the set size
//      if(totalSize != 0) {
//
//        // broadcast the set size change so far
//        PDBCatalogClient pdbClient(getConfiguration()->managerPort, getConfiguration()->managerAddress, logger);
//        pdbClient.incrementKeyRecordInfo(getConfiguration()->getNodeIdentifier(),
//                                         request->databaseName,
//                                         PDBCatalog::fromKeySetNameToSetName(std::string(request->setName)),
//                                         totalSize,
//                                         totalRecords,
//                                         error);
//      }
//
//      // finish
//      return std::make_pair(success, error);
//    }
//
//    // ok we did not freeze the page
//    page->freezeSize(freezeSize);
//
//    // end writing to a page
//    {
//      // lock to do the bookkeeping
//      unique_lock<std::mutex> lck(pageMutex);
//
//      // finish writing to the set
//      endWritingToPage(set, pageNum);
//
//      // decrement the size of the set
//      incrementSetSize(set, freezeSize);
//    }
//
//    // increment the set size
//    totalSize += freezeSize;
//  }
//
//  /// 4. Update the set size
//
//  // broadcast the set size change so far
//  PDBCatalogClient pdbClient(getConfiguration()->managerPort, getConfiguration()->managerAddress, logger);
//  pdbClient.incrementKeyRecordInfo(getConfiguration()->getNodeIdentifier(),
//                                   request->databaseName,
//                                   PDBCatalog::fromKeySetNameToSetName(std::string(request->setName)),
//                                   totalSize,
//                                   totalRecords,
//                                   error);
//
//  /// 5. Finish this, by sending an ack
//
//  // set the response
//  simpleResponse->res = true;
//  simpleResponse->errMsg = error;
//
//  // sends result to requester
//  sendUsingMe->sendObject(simpleResponse, error);
//
//  // if we failed end here
//  if(!success) {
//
//    // return the result
//    return std::make_pair(success, error);
//  }

  // we succeeded
  return std::make_pair(success, error);
}

template <class Communicator>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleRemovePageSet(pdb::Handle<pdb::StoRemovePageSetRequest> &request, std::shared_ptr<Communicator> &sendUsingMe) {

  // the error
  std::string error;

  /// 1. Connect to the backend

  // connect to the backend
  std::shared_ptr<Communicator> communicatorToBackend = make_shared<Communicator>();
  if (!communicatorToBackend->connectToLocalServer(logger, getConfiguration()->ipcFile, error)) {
    return std::make_pair(false, error);
  }

  /// 2. Forward the request

  // sends result to requester
  bool success = communicatorToBackend->sendObject(request, error, 1024);

  // did we succeed in send the stuff
  if(!success) {
    return std::make_pair(false, error);
  }

  /// 4. Wait for response

  // wait for the storage finish result
  success = RequestFactory::waitHeapRequest<SimpleRequestResult, bool>(logger, communicatorToBackend, false,
   [&](Handle<SimpleRequestResult> result) {

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

  /// 5. Send a response

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response
  pdb::Handle<pdb::SimpleRequestResult> simpleResponse = pdb::makeObject<pdb::SimpleRequestResult>(success, error);

  // sends result to requester
  sendUsingMe->sendObject(simpleResponse, error);

  // return
  return std::make_pair(success, error);
}

template <class Communicator>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleClearSetRequest(pdb::Handle<pdb::StoClearSetRequest> &request,
                                                                                   std::shared_ptr<Communicator> &sendUsingMe) {
  std::string error;

  // lock the structures
  std::unique_lock<std::mutex> lck{pageMutex};

  // make the set
  auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

  // remove the stats
  pageStats.erase(set);

  // make sure we have no pages that we are writing to
  auto it = pagesBeingWrittenTo.find(set);
  if(it != pagesBeingWrittenTo.end() && !it->second.empty()) {

    // set the error
    error = "There are currently pages being written to, failed to remove the set.";

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};
    Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, error);

    // sends result to requester
    sendUsingMe->sendObject(response, error);

    // return
    return std::make_pair(false, error);
  }

  // remove the pages being written to
  pagesBeingWrittenTo.erase(set);

  // remove the skipped pages
  freeSkippedPages.erase(set);

  // get the buffer manger
  auto bufferManager = std::dynamic_pointer_cast<pdb::PDBBufferManagerImpl>(getFunctionalityPtr<pdb::PDBBufferManagerInterface>());

  // clear the set from the buffer manager
  bufferManager->clearSet(set);

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};
  Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(true, error);

  // sends result to requester
  sendUsingMe->sendObject(response, error);

  // return
  return std::make_pair(true, error);
}

//template <class Communicator>
//std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleStartFeedingPageSetRequest(pdb::Handle<pdb::StoStartFeedingPageSetRequest> &request,
//                                                                                              std::shared_ptr<Communicator> &sendUsingMe) {
//  bool success = true;
//  std::string error;
//
//  /// 1. Connect to the backend
//
//  // try to connect to the backend
//  int32_t retries = 0;
//  PDBCommunicatorPtr communicatorToBackend = make_shared<PDBCommunicator>();
//  while (!communicatorToBackend->connectToInternetServer(logger, getConfiguration()->port, getConfiguration()->address, error)) {
//
//    // if we are out of retries finish
//    if(retries >= getConfiguration()->maxRetries) {
//      success = false;
//      break;
//    }
//
//    // we used up a retry
//    retries++;
//  }
//
//  // if we could not connect to the backend we failed
//  if(!success) {
//
//    // return
//    return std::make_pair(success, error);
//  }
//
//  /// 2. Forward the request to the backend
//
//  success = communicatorToBackend->sendObject(request, error, 1024);
//
//  // if we succeeded in sending the object, we expect an ack
//  if(success) {
//
//    // create an allocation block to hold the response
//    const UseTemporaryAllocationBlock localBlock{1024};
//
//    // get the next object
//    auto result = communicatorToBackend->template getNextObject<pdb::SimpleRequestResult>(success, error);
//
//    // set the success indicator...
//    success = result != nullptr && result->res;
//  }
//
//  /// 3. Send a response to the other node about the status
//
//  // create an allocation block to hold the response
//  const UseTemporaryAllocationBlock tempBlock{1024};
//
//  // create the response for the other node
//  pdb::Handle<pdb::SimpleRequestResult> simpleResponse = pdb::makeObject<pdb::SimpleRequestResult>(success, error);
//
//  // sends result to requester
//  success = sendUsingMe->sendObject(simpleResponse, error);
//
//  /// 4. If everything went well, start getting the pages from the other node
//
//  // get the buffer manager
//  auto bufferManager = std::dynamic_pointer_cast<pdb::PDBBufferManagerFrontEnd>(getFunctionalityPtr<pdb::PDBBufferManagerInterface>());
//
//  // if everything went well start receiving the pages
//  while(success) {
//
//    /// 4.1 Get the signal that there are more pages
//
//    // create an allocation block to hold the response
//    const UseTemporaryAllocationBlock localBlock{1024};
//
//    // get the next object
//    auto hasPage = sendUsingMe->template getNextObject<pdb::StoFeedPageRequest>(success, error);
//
//    // if we failed break
//    if(!success) {
//      break;
//    }
//
//    /// 4.2 Forward that signal so that the backend knows that there are more pages
//
//    // forward the feed page request
//    success = communicatorToBackend->sendObject<pdb::StoFeedPageRequest>(hasPage, error, 1024);
//
//    // if we failed break
//    if(!success) {
//      break;
//    }
//
//    // do we have a page, if we don't finish
//    if(!hasPage->hasNextPage){
//      break;
//    }
//
//    /// 4.3 Get the page from the other node
//
//    // get the page of the size we need
//    auto page = bufferManager->getPage(hasPage->pageSize);
//
//    // grab the bytes
//    success = sendUsingMe->receiveBytes(page->getBytes(), error);
//
//    // if we failed finish
//    if(!success) {
//      break;
//    }
//
//    /// 4.4 Forward the page to the backend
//
//    // forward the page to the backend
//    success = bufferManager->forwardPage(page, communicatorToBackend, error);
//  }
//
//  // return
//  return std::make_pair(success, error);
//}

template <class Communicator>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleStoreData(const pdb::Handle<pdb::StoStoreDataRequest> &request,
                                                                            std::shared_ptr<Communicator> &sendUsingMe) {

//  /// 1. Grab a page and decompress the forwarded page
//
//  // grab the buffer manager
//  auto bufferManager = this->getFunctionalityPtr<PDBBufferManagerInterface>();
//
//  // grab the forwarded page
//  auto inPage = bufferManager->expectPage(sendUsingMe);
//
//  // check the uncompressed size
//  size_t uncompressedSize = 0;
//  snappy::GetUncompressedLength((char*) inPage->getBytes(), request->compressedSize, &uncompressedSize);
//
//  // grab the page
//  auto outPage = bufferManager->getPage(make_shared<pdb::PDBSet>(request->databaseName, request->setName), request->page);
//
//  // uncompress and copy to page
//  snappy::RawUncompress((char*) inPage->getBytes(), request->compressedSize, (char*) outPage->getBytes());
//
//  // freeze the page
//  outPage->freezeSize(uncompressedSize);
//
//  /// 2. Update the set size
//  {
//    // figure out the number of records
//    Handle<Vector<Handle<Object>>> data = ((Record<Vector<Handle<Object>>> *) outPage->getBytes())->getRootObject();
//    uint64_t numRecords = data->size();
//
//    // send the catalog that data has been added
//    std::string errMsg;
//    PDBCatalogClient pdbClient(getConfiguration()->managerPort, getConfiguration()->managerAddress, logger);
//    if (!pdbClient.incrementSetRecordInfo(getConfiguration()->getNodeIdentifier(),
//                                          request->databaseName,
//                                          request->setName,
//                                          uncompressedSize,
//                                          numRecords,
//                                          errMsg)) {
//
//      // create an allocation block to hold the response
//      const UseTemporaryAllocationBlock tempBlock{1024};
//      Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);
//
//      // sends result to requester
//      sendUsingMe->sendObject(response, errMsg);
//      return make_pair(false, errMsg);
//    }
//  }
//
//  /// 3. Send the response that we are done
//
//  // create an allocation block to hold the response
//  string error;
//  pdb::Handle<pdb::SimpleRequestResult> simpleResponse = pdb::makeObject<pdb::SimpleRequestResult>(true, error);
//
//  // sends result to requester
//  sendUsingMe->sendObject(simpleResponse, error);

  // finish
  std::string error;
  return make_pair(true, error);
}

template <class Communicator>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleStoreKeys(const pdb::Handle<pdb::StoStoreKeysRequest> &request,
                                                                            std::shared_ptr<Communicator> &sendUsingMe) {

//  /// 1. Grab a page and decompress the forwarded page
//
//  // grab the buffer manager
//  auto bufferManager = this->getFunctionalityPtr<PDBBufferManagerInterface>();
//
//  // grab the forwarded page
//  auto inPage = bufferManager->expectPage(sendUsingMe);
//
//  // check the uncompressed size
//  size_t uncompressedSize = 0;
//  snappy::GetUncompressedLength((char*) inPage->getBytes(), request->compressedSize, &uncompressedSize);
//
//  // grab the page
//  auto outPage = bufferManager->getPage(make_shared<pdb::PDBSet>(request->databaseName, request->setName), request->page);
//
//  // uncompress and copy to page
//  snappy::RawUncompress((char*) inPage->getBytes(), request->compressedSize, (char*) outPage->getBytes());
//
//  // freeze the page
//  outPage->freezeSize(uncompressedSize);
//
//  /// 3. Update the set size
//  {
//    // cast the place where we copied the thing
//    auto* recordCopy = (Record<Vector<Handle<Object>>>*) outPage->getBytes();
//
//    // grab the copy of the supervisor object
//    Handle<Vector<Handle<Object>>> keyVector = recordCopy->getRootObject();
//
//    // send the catalog that data has been added
//    std::string errMsg;
//    PDBCatalogClient pdbClient(getConfiguration()->managerPort, getConfiguration()->managerAddress, logger);
//    if (!pdbClient.incrementKeyRecordInfo(getConfiguration()->getNodeIdentifier(),
//                                          request->databaseName,
//                                          PDBCatalog::fromKeySetNameToSetName(std::string(request->setName)),
//                                          uncompressedSize,
//                                          keyVector->size(),
//                                          errMsg)) {
//
//      // create an allocation block to hold the response
//      const UseTemporaryAllocationBlock tempBlock{1024};
//      Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);
//
//      // sends result to requester
//      sendUsingMe->sendObject(response, errMsg);
//      return make_pair(false, errMsg);
//    }
//  }
//
//  /// 3. Send the response that we are done
//
//  // create an allocation block to hold the response
//  string error;
//  pdb::Handle<pdb::SimpleRequestResult> simpleResponse = pdb::makeObject<pdb::SimpleRequestResult>(true, error);
//
//  // sends result to requester
//  sendUsingMe->sendObject(simpleResponse, error);

  // finish
  std::string error;
  return make_pair(true, error);
}

template<class Communicator>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handlePageSet(const pdb::Handle<pdb::StoRemovePageSetRequest> &request, shared_ptr<Communicator> &sendUsingMe) {

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

  // return success
  return make_pair(success, error);
}

template<class Communicator>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleStartFeedingPageSetRequest(pdb::Handle<pdb::StoStartFeedingPageSetRequest> &request,
                                                                                             shared_ptr<Communicator> &sendUsingMe) {
  std::string error;
  bool success;
//
//  /// 1. First grab the page set we are going to feed if it does not exist it will be created
//
//  // create or grab the page set
//  auto pageSet = createFeedingAnonymousPageSet(request->getPageSetID(), request->numberOfProcessingThreads, request->numberOfNodes);
//
//  // if we got the page success is true
//  success = pageSet != nullptr;
//
//  /// 2. Next we send a signal that we have acknowledged the request
//
//  // create an allocation block to hold the response
//  const UseTemporaryAllocationBlock tempBlock{1024};
//
//  // create the response for the other node
//  pdb::Handle<pdb::SimpleRequestResult> simpleResponse = pdb::makeObject<pdb::SimpleRequestResult>(success, error);
//
//  // sends result to requester
//  success = sendUsingMe->sendObject(simpleResponse, error);
//
//  /// 3. Start receiving the pages
//
//  // get the buffer manager
//  auto bufferManager = getFunctionalityPtr<pdb::PDBBufferManagerInterface>();
//
//  while(success) {
//
//    /// 3.1 Get the info about the next page if any
//
//    // create an allocation block to hold the response
//    const UseTemporaryAllocationBlock localBlock{1024};
//
//    // get the next object
//    auto hasPage = sendUsingMe->template getNextObject<pdb::StoFeedPageRequest>(success, error);
//
//    // if we failed break
//    if(!success) {
//      break;
//    }
//
//    // if we don't have a page
//    if(!hasPage->hasNextPage) {
//
//      // this is a regular exit
//      success = true;
//      break;
//    }
//
//    /// 3.2 Grab the forwarded page
//
//    // get the page from the frontend
//    auto page = bufferManager->expectPage(sendUsingMe);
//
//    // if we did not get a page break something is wrong
//    if(page == nullptr) {
//      success = false;
//      break;
//    }
//
//    // unpin the page
//    page->unpin();
//
//    // feed the page to the page set
//    pageSet->feedPage(page);
//
//    // we were successful in feeding the page set
//    success = true;
//  }
//
//  // if we got a page set mark that we are finished feeding
//  if(pageSet != nullptr) {
//
//    // finish feeding
//    pageSet->finishFeeding();
//  }

  // return
  return std::make_pair(success, error);
}

#endif //PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_H
