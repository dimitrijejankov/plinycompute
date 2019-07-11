//
// Created by dimitrije on 2/17/19.
//

#ifndef PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_H
#define PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_H

#include <PDBStorageManagerFrontend.h>
#include <HeapRequestHandler.h>
#include <StoDispatchData.h>
#include <PDBBufferManagerInterface.h>
#include <PDBBufferManagerFrontEnd.h>
#include <StoStoreOnPageRequest.h>
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
  auto maxCompressedSize = std::min<size_t>(snappy::MaxCompressedLength(pageRecord->numBytes()), 128 * 1024 * 1024);
  auto compressedPage = getFunctionalityPtr<PDBBufferManagerInterface>()->getPage(maxCompressedSize);

  // compress the record
  size_t compressedSize;
  snappy::RawCompress((char*) pageRecord, pageRecord->numBytes(), (char*)compressedPage->getBytes(), &compressedSize);

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

template <class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleDispatchedData(pdb::Handle<pdb::StoDispatchData> request, std::shared_ptr<Communicator> sendUsingMe) {

  /// 1. Get the page from the distributed storage

  // the error
  std::string error;

  // grab the buffer manager
  auto bufferManager = std::dynamic_pointer_cast<pdb::PDBBufferManagerFrontEnd>(getFunctionalityPtr<pdb::PDBBufferManagerInterface>());

  // figure out how large the compressed payload is
  size_t numBytes = sendUsingMe->getSizeOfNextObject();

  // grab a page to write this
  auto page = bufferManager->getPage(numBytes);

  // grab the bytes
  auto success = sendUsingMe->receiveBytes(page->getBytes(), error);

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
  snappy::GetUncompressedLength((char*) page->getBytes(), numBytes, &uncompressedSize);

  /// 2. Figure out the page we want to put this thing onto

  uint64_t pageNum;
  {
    // lock the stuff that keeps track of the last page
    unique_lock<std::mutex> lck(pageMutex);

    // make the set
    auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

    // get the next page
    pageNum = getNextFreePage(set);

    // indicate that we are writing to this page
    startWritingToPage(set, pageNum);

    // increment the set size
    incrementSetSize(set, uncompressedSize);
  }

  /// 3. Initiate the storing on the backend

  PDBCommunicatorPtr communicatorToBackend = make_shared<PDBCommunicator>();
  if (!communicatorToBackend->connectToLocalServer(logger, getConfiguration()->ipcFile, error)) {
    return std::make_pair(false, error);
  }

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};
  Handle<StoStoreOnPageRequest> response = makeObject<StoStoreOnPageRequest>(request->databaseName, request->setName, pageNum, request->compressedSize);

  // send the thing to the backend
  if (!communicatorToBackend->sendObject(response, error)) {

    // make the set
    auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

    // set the indicators and send a NACK to the client since we failed
    handleDispatchFailure(set, pageNum, uncompressedSize, sendUsingMe);

    // finish
    return std::make_pair(false, error);
  }

  /// 4. Forward the page to the backend

  // forward the page
  if(!bufferManager->forwardPage(page, communicatorToBackend, error)) {

    // we could not forward the page
    auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

    // set the indicators and send a NACK to the client since we failed
    handleDispatchFailure(set, pageNum, uncompressedSize, sendUsingMe);

    // finish
    return std::make_pair(false, error);
  }

  /// 5. Wait for the backend to finish the stuff

  success = Requests::template waitHeapRequest<SimpleRequestResult, bool>(logger, communicatorToBackend, false,
  [&](Handle<SimpleRequestResult> result) {

   // check the result
   if (result != nullptr && result->getRes().first) {
     return true;
   }

   // since we failed just invalidate the set
   auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);
   {
    // lock the stuff
    unique_lock<std::mutex> lck(pageMutex);

    // finish writing to the set
    endWritingToPage(set, pageNum);

    // return the page to the free list
    freeSetPage(set, pageNum);

    // decrement the size of the set
    decrementSetSize(set, uncompressedSize);
   }

   // log the error
   error = "Error response from distributed-storage: " + result->getRes().second;
   logger->error("Error response from distributed-storage: " + result->getRes().second);

   return false;
  });

  // finish writing to the set
  endWritingToPage(std::make_shared<PDBSet>(request->databaseName, request->setName), pageNum);

  /// 6. Send the response that we are done

  // create an allocation block to hold the response
  Handle<SimpleRequestResult> simpleResponse = makeObject<SimpleRequestResult>(success, error);

  // sends result to requester
  success = sendUsingMe->sendObject(simpleResponse, error) && success;

  // finish
  return std::make_pair(success, error);
}

template<class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleGetSetPages(pdb::Handle<pdb::StoGetSetPagesRequest> request, shared_ptr<Communicator> sendUsingMe) {

  // the error
  std::string error;
  bool success;

  // check if the set exists
  if(!getFunctionalityPtr<pdb::PDBCatalogClient>()->setExists(request->databaseName, request->setName)) {

    // set the error
    error = "The set the pages were requested does not exist!";
    success = false;

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
  auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

  std::vector<uint64_t> pages;
  {
    // lock the stuff
    unique_lock<std::mutex> lck(pageMutex);

    // try to find the page
    auto it = this->pageStats.find(set);
    if(it != this->pageStats.end()) {

      // reserve the pages
      pages.reserve(it->second.lastPage);

      // do we even have this page
      uint64_t currPage = 0;
      while(currPage <= it->second.lastPage) {

        // check if the page is valid
        if(pageExists(set, currPage) && !isPageBeingWrittenTo(set, currPage) && !isPageFree(set, currPage)) {
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
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleMaterializeSet(pdb::Handle<pdb::StoMaterializePageSetRequest> request,
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
    success = true;
  }

  /// 2. send an ACK or NACK depending on whether the set exists

  // make an allocation block
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create an allocation block to hold the response
  Handle<SimpleRequestResult> simpleResponse = makeObject<SimpleRequestResult>(success, error);

  // sends result to requester
  sendUsingMe->sendObject(simpleResponse, error);

  // if we failed end here
  if(!success) {

    // return the result
    return std::make_pair(success, error);
  }

  /// 3. Send pages over the wire to the backend

  // grab the buffer manager
  auto bufferManager = std::dynamic_pointer_cast<pdb::PDBBufferManagerFrontEnd>(getFunctionalityPtr<pdb::PDBBufferManagerInterface>());

  // make the set
  auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

  // this is going to count the total size of the pages
  uint64_t totalSize = 0;

  // start forwarding the pages
  bool hasNext = true;
  while (hasNext) {

    uint64_t pageNum;
    {
      // lock to do the bookkeeping
      unique_lock<std::mutex> lck(pageMutex);

      // get the next page
      pageNum = getNextFreePage(set);

      // indicate that we are writing to this page
      startWritingToPage(set, pageNum);
    }

    // get the page
    auto page = bufferManager->getPage(set, pageNum);

    // forward the page to the backend
    success = bufferManager->forwardPage(page, sendUsingMe, error);

    // did we fail?
    if(!success) {

      // lock to do the bookkeeping
      unique_lock<std::mutex> lck(pageMutex);

      // finish writing to the set
      endWritingToPage(set, pageNum);

      // return the page to the free list
      freeSetPage(set, pageNum);

      // free the lock
      lck.unlock();

      // do we need to update the set size
      if(totalSize != 0) {

        // broadcast the set size change so far
        this->getFunctionalityPtr<PDBCatalogClient>()->incrementSetSize(set->getDBName(), set->getSetName(), totalSize, error);
      }

      // finish here since this is not recoverable on the backend
      return std::make_pair(success, "Error occurred while forwarding the page to the backend.\n" + error);
    }

    // the size we want to freeze this thing to
    size_t freezeSize = 0;

    // wait for the storage finish result
    success = RequestFactory::waitHeapRequest<StoMaterializePageResult, bool>(logger, sendUsingMe, false,
      [&](Handle<StoMaterializePageResult> result) {

        // check the result
        if (result != nullptr && result->success) {

          // set the freeze size
          freezeSize = result->materializeSize;

          // set the has next
          hasNext = result->hasNext;

          // finish
          return result->success;
        }

        // set the error
        error = "Backend materializing the page failed!";

        // we failed return so
        return false;
      });

    // did we fail?
    if(!success) {

      // lock to do the bookkeeping
      unique_lock<std::mutex> lck(pageMutex);

      // finish writing to the set
      endWritingToPage(set, pageNum);

      // return the page to the free list
      freeSetPage(set, pageNum);

      // free the lock
      lck.unlock();

      // do we need to update the set size
      if(totalSize != 0) {

        // broadcast the set size change so far
        this->getFunctionalityPtr<PDBCatalogClient>()->incrementSetSize(set->getDBName(), set->getSetName(), totalSize, error);
      }

      // finish
      return std::make_pair(success, error);
    }

    // ok we did not freeze the page
    page->freezeSize(freezeSize);

    // end writing to a page
    {
      // lock to do the bookkeeping
      unique_lock<std::mutex> lck(pageMutex);

      // finish writing to the set
      endWritingToPage(set, pageNum);

      // decrement the size of the set
      incrementSetSize(set, freezeSize);
    }

    // increment the set size
    totalSize += freezeSize;
  }

  /// 4. Update the set size

  // broadcast the set size change so far
  success = this->getFunctionalityPtr<PDBCatalogClient>()->incrementSetSize(set->getDBName(), set->getSetName(), totalSize, error);

  /// 5. Finish this

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
  auto bufferManager = std::dynamic_pointer_cast<pdb::PDBBufferManagerFrontEnd>(getFunctionalityPtr<pdb::PDBBufferManagerInterface>());

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

template <class Communicator>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleStartFeedingPageSetRequest(pdb::Handle<pdb::StoStartFeedingPageSetRequest> &request,
                                                                                              std::shared_ptr<Communicator> &sendUsingMe) {
  bool success = true;
  std::string error;

  /// 1. Connect to the backend

  // try to connect to the backend
  int32_t retries = 0;
  PDBCommunicatorPtr communicatorToBackend = make_shared<PDBCommunicator>();
  while (!communicatorToBackend->connectToLocalServer(logger, getConfiguration()->ipcFile, error)) {

    // if we are out of retries finish
    if(retries >= getConfiguration()->maxRetries) {
      success = false;
      break;
    }

    // we used up a retry
    retries++;
  }

  // if we could not connect to the backend we failed
  if(!success) {

    // return
    return std::make_pair(success, error);
  }

  /// 2. Forward the request to the backend

  success = communicatorToBackend->sendObject(request, error, 1024);

  // if we succeeded in sending the object, we expect an ack
  if(success) {

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock localBlock{1024};

    // get the next object
    auto result = communicatorToBackend->template getNextObject<pdb::SimpleRequestResult>(success, error);

    // set the success indicator...
    success = result != nullptr && result->res;
  }

  /// 3. Send a response to the other node about the status

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response for the other node
  pdb::Handle<pdb::SimpleRequestResult> simpleResponse = pdb::makeObject<pdb::SimpleRequestResult>(success, error);

  // sends result to requester
  success = sendUsingMe->sendObject(simpleResponse, error);

  /// 4. If everything went well, start getting the pages from the other node

  // get the buffer manager
  auto bufferManager = std::dynamic_pointer_cast<pdb::PDBBufferManagerFrontEnd>(getFunctionalityPtr<pdb::PDBBufferManagerInterface>());

  // if everything went well start receiving the pages
  while(success) {

    /// 4.1 Get the signal that there are more pages

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock localBlock{1024};

    // get the next object
    auto hasPage = sendUsingMe->template getNextObject<pdb::StoFeedPageRequest>(success, error);

    // if we failed break
    if(!success) {
      break;
    }

    /// 4.2 Forward that signal so that the backend knows that there are more pages

    // forward the feed page request
    success = communicatorToBackend->sendObject<pdb::StoFeedPageRequest>(hasPage, error, 1024);

    // if we failed break
    if(!success) {
      break;
    }

    // do we have a page, if we don't finish
    if(!hasPage->hasNextPage){
      break;
    }

    /// 4.3 Get the page from the other node

    // get the page of the size we need
    auto page = bufferManager->getPage(hasPage->pageSize);

    // grab the bytes
    success = sendUsingMe->receiveBytes(page->getBytes(), error);

    // if we failed finish
    if(!success) {
      break;
    }

    /// 4.4 Forward the page to the backend

    // forward the page to the backend
    success = bufferManager->forwardPage(page, communicatorToBackend, error);
  }

  // return
  return std::make_pair(success, error);
}

#endif //PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_H
