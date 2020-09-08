#pragma once

#include <ServerFunctionality.h>
#include <StoStoreDataRequest.h>
#include <PDBStorageManagerBackend.h>
#include <PDBBufferManagerBackEnd.h>
#include <StoFeedPageRequest.h>
#include <PDBBufferManagerDebugBackEnd.h>
#include <StoStoreKeysRequest.h>
#include <PDBCatalogClient.h>
#include <PDBCatalog.h>

template <class Communicator>
std::pair<bool, std::string> pdb::PDBStorageManagerBackend::handleStoreData(const pdb::Handle<pdb::StoStoreDataRequest> &request,
                                                                            std::shared_ptr<Communicator> &sendUsingMe) {

  /// 1. Grab a page and decompress the forwarded page

  // grab the buffer manager
  auto bufferManager = std::dynamic_pointer_cast<pdb::PDBBufferManagerBackEndImpl>(this->getFunctionalityPtr<PDBBufferManagerInterface>());

  // grab the forwarded page
  auto inPage = bufferManager->expectPage(sendUsingMe);

  // check the uncompressed size
  size_t uncompressedSize = 0;
  snappy::GetUncompressedLength((char*) inPage->getBytes(), request->compressedSize, &uncompressedSize);

  // grab the page
  auto outPage = bufferManager->getPage(make_shared<pdb::PDBSet>(request->databaseName, request->setName), request->page);

  // uncompress and copy to page
  snappy::RawUncompress((char*) inPage->getBytes(), request->compressedSize, (char*) outPage->getBytes());

  // freeze the page
  outPage->freezeSize(uncompressedSize);

  /// 2. Update the set size
  {
    // figure out the number of records
    Handle<Vector<Handle<Object>>> data = ((Record<Vector<Handle<Object>>> *) outPage->getBytes())->getRootObject();
    uint64_t numRecords = data->size();

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

  /// 3. Send the response that we are done

  // create an allocation block to hold the response
  string error;
  pdb::Handle<pdb::SimpleRequestResult> simpleResponse = pdb::makeObject<pdb::SimpleRequestResult>(true, error);

  // sends result to requester
  sendUsingMe->sendObject(simpleResponse, error);

  // finish
  return make_pair(true, error);
}

template <class Communicator>
std::pair<bool, std::string> pdb::PDBStorageManagerBackend::handleStoreKeys(const pdb::Handle<pdb::StoStoreKeysRequest> &request,
                                                                            std::shared_ptr<Communicator> &sendUsingMe) {

  /// 1. Grab a page and decompress the forwarded page

  // grab the buffer manager
  auto bufferManager = std::dynamic_pointer_cast<pdb::PDBBufferManagerBackEndImpl>(this->getFunctionalityPtr<PDBBufferManagerInterface>());

  // grab the forwarded page
  auto inPage = bufferManager->expectPage(sendUsingMe);

  // check the uncompressed size
  size_t uncompressedSize = 0;
  snappy::GetUncompressedLength((char*) inPage->getBytes(), request->compressedSize, &uncompressedSize);

  // grab the page
  auto outPage = bufferManager->getPage(make_shared<pdb::PDBSet>(request->databaseName, request->setName), request->page);

  // uncompress and copy to page
  snappy::RawUncompress((char*) inPage->getBytes(), request->compressedSize, (char*) outPage->getBytes());

  // freeze the page
  outPage->freezeSize(uncompressedSize);

  /// 3. Update the set size
  {
    // cast the place where we copied the thing
    auto* recordCopy = (Record<Vector<Handle<Object>>>*) outPage->getBytes();

    // grab the copy of the supervisor object
    Handle<Vector<Handle<Object>>> keyVector = recordCopy->getRootObject();

    // send the catalog that data has been added
    std::string errMsg;
    PDBCatalogClient pdbClient(getConfiguration()->managerPort, getConfiguration()->managerAddress, logger);
    if (!pdbClient.incrementKeyRecordInfo(getConfiguration()->getNodeIdentifier(),
                                          request->databaseName,
                                          PDBCatalog::fromKeySetNameToSetName(std::string(request->setName)),
                                          uncompressedSize,
                                          keyVector->size(),
                                          errMsg)) {

      // create an allocation block to hold the response
      const UseTemporaryAllocationBlock tempBlock{1024};
      Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

      // sends result to requester
      sendUsingMe->sendObject(response, errMsg);
      return make_pair(false, errMsg);
    }
  }

  /// 3. Send the response that we are done

  // create an allocation block to hold the response
  string error;
  pdb::Handle<pdb::SimpleRequestResult> simpleResponse = pdb::makeObject<pdb::SimpleRequestResult>(true, error);

  // sends result to requester
  sendUsingMe->sendObject(simpleResponse, error);

  // finish
  return make_pair(true, error);
}

template<class Communicator>
std::pair<bool, std::string> pdb::PDBStorageManagerBackend::handlePageSet(const pdb::Handle<pdb::StoRemovePageSetRequest> &request, shared_ptr<Communicator> &sendUsingMe) {

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
std::pair<bool, std::string> pdb::PDBStorageManagerBackend::handleStartFeedingPageSetRequest(pdb::Handle<pdb::StoStartFeedingPageSetRequest> &request,
                                                                                             shared_ptr<Communicator> &sendUsingMe) {
  std::string error;
  bool success;

  /// 1. First grab the page set we are going to feed if it does not exist it will be created

  // create or grab the page set
  auto pageSet = createFeedingAnonymousPageSet(request->getPageSetID(), request->numberOfProcessingThreads, request->numberOfNodes);

  // if we got the page success is true
  success = pageSet != nullptr;

  /// 2. Next we send a signal that we have acknowledged the request

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response for the other node
  pdb::Handle<pdb::SimpleRequestResult> simpleResponse = pdb::makeObject<pdb::SimpleRequestResult>(success, error);

  // sends result to requester
  success = sendUsingMe->sendObject(simpleResponse, error);

  /// 3. Start receiving the pages

  // get the buffer manager
  auto bufferManager = std::dynamic_pointer_cast<pdb::PDBBufferManagerBackEndImpl>(getFunctionalityPtr<pdb::PDBBufferManagerInterface>());

  while(success) {

    /// 3.1 Get the info about the next page if any

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock localBlock{1024};

    // get the next object
    auto hasPage = sendUsingMe->template getNextObject<pdb::StoFeedPageRequest>(success, error);

    // if we failed break
    if(!success) {
      break;
    }

    // if we don't have a page
    if(!hasPage->hasNextPage) {

      // this is a regular exit
      success = true;
      break;
    }

    /// 3.2 Grab the forwarded page

    // get the page from the frontend
    auto page = bufferManager->expectPage(sendUsingMe);

    // if we did not get a page break something is wrong
    if(page == nullptr) {
      success = false;
      break;
    }

    // unpin the page
    page->unpin();

    // feed the page to the page set
    pageSet->feedPage(page);

    // we were successful in feeding the page set
    success = true;
  }

  // if we got a page set mark that we are finished feeding
  if(pageSet != nullptr) {

    // finish feeding
    pageSet->finishFeeding();
  }

  // return
  return std::make_pair(success, error);
}