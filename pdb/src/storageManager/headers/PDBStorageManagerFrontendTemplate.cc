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

template <class T>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleGetPageRequest(const pdb::Handle<pdb::StoGetPageRequest> &request,
                                                                                  std::shared_ptr<T>  &sendUsingMe) {


  /// 1. Check if we have the page

  // create the set identifier
  auto set = make_shared<pdb::PDBSet>(request->databaseName, request->setName);

  bool hasPage = false;
  // check if the set exists
  {
    // lock the stuff that keeps track of the last page
    unique_lock<mutex> lck;

    // check for the page
    auto it = this->lastPages.find(set);
    if(it != this->lastPages.end() && it->second >= request->page) {
      hasPage = true;
    }
  }

  /// 2. If we don't have it send an error

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

  /// 3. We grab the page and compress it.

  // grab the page
  auto page = this->getFunctionalityPtr<PDBBufferManagerInterface>()->getPage(set, request->page);

  // grab the vector
  auto* pageRecord = (pdb::Record<pdb::Vector<pdb::Handle<pdb::Object>>> *) (page->getBytes());

  // grab an anonymous page to store the compressed stuff
  auto maxCompressedSize = snappy::MaxCompressedLength(pageRecord->numBytes());
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

  /// 2. Figure out the page we want to put this thing onto

  uint64_t pageNum;
  {
    // lock the stuff that keeps track of the last page
    unique_lock<std::mutex> lck(m);

    // make the set
    auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

    // try to find the set
    auto it = lastPages.find(set);

    // do we even have a record for this set
    if(it == lastPages.end()) {

      // set the page to zero since this is the first page
      lastPages[set] = 0;
      pageNum = 0;
    }
    else {

      // increment the last page
      pageNum = ++it->second;
    }
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

    // finish
    return std::make_pair(false, std::string("Could not send the thing to the backend"));
  }

  /// 4. Forward the page to the backend

  // forward the page
  success = bufferManager->forwardPage(page, communicatorToBackend, error);

  /// 5. Wait for the backend to finish the stuff

  Requests::template waitHeapRequest<SimpleRequestResult, bool>(logger, communicatorToBackend, false,
  [&](Handle<SimpleRequestResult> result) {

   // check the result
   if (result != nullptr && result->getRes().first) {
     return true;
   }

   // log the error
   error = "Error response from distributed-storage: " + result->getRes().second;
   logger->error("Error response from distributed-storage: " + result->getRes().second);

   return false;
  });

  /// 6. Send the response that we are done

  // create an allocation block to hold the response
  Handle<SimpleRequestResult> simpleResponse = makeObject<SimpleRequestResult>(true, error);

  // sends result to requester
  sendUsingMe->sendObject(simpleResponse, error);

  // finish
  return std::make_pair(success, error);
}

template<class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontend::handleGetNumPages(pdb::Handle<pdb::StoSetStatsRequest> request, shared_ptr<Communicator> sendUsingMe) {

  // the error
  std::string error;
  bool success = true;

  // make the set identifier
  auto set = std::make_shared<PDBSet>(request->databaseName, request->setName);

  // try to find the set
  auto it = lastPages.find(set);

  const UseTemporaryAllocationBlock tempBlock{1024};

  // figure out whether this was a success
  pdb::Handle<pdb::StoSetStatsResult> setStatResult;
  if(it != lastPages.end()) {

    // TODO I need to keep track of the set sizes
    setStatResult = pdb::makeObject<pdb::StoSetStatsResult>(it->second, 0, true);
  }
  else {

    // we failed
    setStatResult = pdb::makeObject<pdb::StoSetStatsResult>(0, 0, false);
    success = false;
  }

  // sends result to requester
  success = success && sendUsingMe->sendObject(setStatResult, error);

  // return the result
  return std::make_pair(success, error);
}

#endif //PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_H
