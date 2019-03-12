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
#include <StoSetStatsResult.h>
#include <StoStartWritingToSetResult.h>

template <class T>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleGetPageRequest(const pdb::Handle<pdb::StoGetPageRequest> &request,
                                                                                  std::shared_ptr<T>  &sendUsingMe) {


  /// 1. Check if we have the page

  // create the set identifier
  auto set = make_shared<pdb::PDBSet>(request->databaseName, request->setName);

  bool hasPage;
  // check if the page exists exists
  {
    // lock the stuff that keeps track of the last page
    unique_lock<mutex> lck;

    // check for the page exists and it is in a valid state
    hasPage = pageExists(set, request->page) && !isPageBeingWrittenTo(set, request->page) && !isPageFree(set, request->page);
  }

  /// 2. If we don't have it or it is not in a valid state it send back a NACK

  if(!hasPage) {

    // make an allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{1024};

    // create an allocation block to hold the response
    pdb::Handle<pdb::StoGetPageResult> response = pdb::makeObject<pdb::StoGetPageResult>(0, false);

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
  auto maxCompressedSize = std::min<size_t>(snappy::MaxCompressedLength(pageRecord->numBytes()), 128 * 1024 * 1024);
  auto compressedPage = getFunctionalityPtr<PDBBufferManagerInterface>()->getPage(maxCompressedSize);

  // compress the record
  size_t compressedSize;
  snappy::RawCompress((char*) pageRecord, pageRecord->numBytes(), (char*)compressedPage->getBytes(), &compressedSize);

  /// 4. Send the compressed page

  // make an allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{1024};

  // create an allocation block to hold the response
  pdb::Handle<pdb::StoGetPageResult> response = pdb::makeObject<pdb::StoGetPageResult>(compressedSize, true);

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
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleDispatchedData(pdb::Handle<pdb::StoDispatchData> request, std::shared_ptr<Communicator> sendUsingMe)  {

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
  if (communicatorToBackend->connectToLocalServer(logger, getConfiguration()->ipcFile, error)) {

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
    handleDispatchFailure(set, pageNum, sendUsingMe);

    // finish
    return std::make_pair(false, error);
  }

  /// 4. Forward the page to the backend

  // forward the page
  if(!bufferManager->forwardPage(page, communicatorToBackend, error)) {

    // we could not forward the page
    auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

    // set the indicators and send a NACK to the client since we failed
    handleDispatchFailure(set, pageNum, sendUsingMe);

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

   // just invalidate the set
   auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);
   {
    // lock the stuff
    unique_lock<std::mutex> lck(pageMutex);

    // finish writing to the set
    endWritingToPage(set, pageNum);

    // return the page to the free list
    freeSetPage(set, pageNum);
   }

   // log the error
   error = "Error response from distributed-storage: " + result->getRes().second;
   logger->error("Error response from distributed-storage: " + result->getRes().second);

   return false;
  });

  /// 6. Freeze the page since it is safe to do now

  page->freezeSize(uncompressedSize);

  /// 7. Send the response that we are done

  // create an allocation block to hold the response
  Handle<SimpleRequestResult> simpleResponse = makeObject<SimpleRequestResult>(success, error);

  // sends result to requester
  success = sendUsingMe->sendObject(simpleResponse, error) && success;

  // finish
  return std::make_pair(success, error);
}

template<class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleGetSetStats(pdb::Handle<pdb::StoSetStatsRequest> request,
                                                                               shared_ptr<Communicator> sendUsingMe) {

  // the error
  std::string error;
  bool success = true;

  // make the set identifier
  auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

  // try to find the set
  auto it = pageStats.find(set);

  const UseTemporaryAllocationBlock tempBlock{1024};

  // figure out whether this was a success
  pdb::Handle<pdb::StoSetStatsResult> setStatResult;
  if(it != pageStats.end()) {

    // set the stat results
    setStatResult = pdb::makeObject<pdb::StoSetStatsResult>(it->second.lastPage + 1, it->second.size, true);
  }
  else {

    // we failed
    setStatResult = pdb::makeObject<pdb::StoSetStatsResult>(0, 0, false);
    success = false;
  }

  // sends result to requester
  success = sendUsingMe->sendObject(setStatResult, error) && success;

  // return the result
  return std::make_pair(success, error);
}

template<class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleStartWritingToSet(pdb::Handle<pdb::StoStartWritingToSetRequest> request, shared_ptr<Communicator> sendUsingMe) {

  /// TODO this has to be more robust, right now this is just here to do the job!

  // make an allocation block
  const UseTemporaryAllocationBlock tempBlock{1024};

  bool success = false;
  uint64_t firstPage = 0;

  // check if the set exists
  if(getFunctionalityPtr<pdb::PDBCatalogClient>()->setExists(request->databaseName, request->setName)) {

    // if it exists do the bookkeeping, lock the stuff that keeps track of the last page
    unique_lock<std::mutex> lck(pageMutex);

    // make the set
    auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

    // try to find the set
    auto it = pageStats.find(set);

    // do we even have a record for this set
    if (it != pageStats.end()) {

      // figure out the page and increment them
      firstPage = it->second.lastPage;
      it->second.lastPage += request->numPages;
    } else {

      // set the page to the (requested number - 1) since the pages start from 0
      pageStats[set].lastPage = request->numPages - 1;

      // set the size to the size of this request
      pageStats[set].size = 0;
      firstPage = 0;
    }

    // ok we managed to find it and get some pages
    success = true;
  }

  // create a response
  Handle<StoStartWritingToSetResult> response = makeObject<StoStartWritingToSetResult>(firstPage, success);

  // send the thing to the backend
  std::string error;
  if (!sendUsingMe->sendObject(response, error)) {

    // finish this
    return std::make_pair(false, std::string("Could not send the thing to the backend"));
  }

  // we succeeded
  return std::make_pair(success, "");
}

#endif //PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_H
