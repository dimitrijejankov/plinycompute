//
// Created by dimitrije on 1/21/19.
//

#ifndef PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_CC
#define PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_CC

#include <SimpleRequestResult.h>
#include <BufPinPageResult.h>
#include <BufGetPageResult.h>
#include <BufFreezeRequestResult.h>
#include <BufForwardPageRequest.h>
#include "assert.h"

template <class T>
std::pair<bool, std::string> pdb::PDBBufferManagerFrontEnd::handleGetPageRequest(pdb::Handle<pdb::BufGetPageRequest> &request, std::shared_ptr<T> &sendUsingMe) {

  // grab the page
  auto page = this->getPage(make_shared<pdb::PDBSet>(request->dbName, request->setName), request->pageNumber);

  // send the page to the backend
  string error;
  bool res = this->sendPageToBackend(page, sendUsingMe, error);

  return make_pair(res, error);
}

template <class T>
std::pair<bool, std::string> pdb::PDBBufferManagerFrontEnd::handleGetAnonymousPageRequest(pdb::Handle<pdb::BufGetAnonymousPageRequest> &request, std::shared_ptr<T> &sendUsingMe) {

  // grab an anonymous page
  auto page = getPage(request->size);

  // send the page to the backend
  std::string error;
  bool res = sendPageToBackend(page, sendUsingMe, error);

  return make_pair(res, error);
}

template <class T>
std::pair<bool, std::string> pdb::PDBBufferManagerFrontEnd::handleReturnPageRequest(pdb::Handle<pdb::BufReturnPageRequest> &request, std::shared_ptr<T> &sendUsingMe) {

  // create the page key
  auto key = std::make_pair(std::make_shared<PDBSet>(request->databaseName, request->setName), request->pageNumber);

  // do the bookkeeping of the sent pages
  bool res = false;
  {
    // lock to do it in a thread safe manner
    unique_lock<mutex> lck(m);

    // try find the page
    auto it = this->sentPages.find(key);

    // did we find it
    if(it != this->sentPages.end()) {

      // mark it as dirty
      it->second->setDirty();

      // are we currently forwarding this page if we are not
      if(this->forwarding.find(key) == this->forwarding.end()) {

        // remove it
        this->sentPages.erase(it);
      }

      // ok this request is a success
      res = true;
    }
  }

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response
  Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(res, res ? std::string("") : std::string("Could not find the page to remove!"));

  // sends result to requester
  std::string errMsg;
  res = sendUsingMe->sendObject(response, errMsg) && res;

  // return
  return make_pair(res, errMsg);
}

template <class T>
std::pair<bool, std::string> pdb::PDBBufferManagerFrontEnd::handleReturnAnonPageRequest(pdb::Handle<pdb::BufReturnAnonPageRequest> &request, std::shared_ptr<T> &sendUsingMe) {

  // create the page key
  auto key = std::make_pair((PDBSetPtr) nullptr, request->pageNumber);

  // remove the anonymous page
  bool res;
  {
    // lock the thing
    unique_lock<mutex> lck(m);

    // find the page
    auto it = this->sentPages.find(key);

    // set the dirty bit
    if(request->isDirty) {
      it->second->isDirty();
    }

    // did we find it
    res = it != this->sentPages.end();

    // are we currently forwarding this page if we are not
    if(this->forwarding.find(key) == this->forwarding.end()) {

      // remove it
      this->sentPages.erase(it);
    }
  }

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response
  Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(res, res ? std::string("") : std::string("Could not find the page to remove!"));

  // sends result to requester
  std::string errMsg;
  res = sendUsingMe->sendObject(response, errMsg) && res;

  // return
  return make_pair(res, errMsg);
}

template <class T>
std::pair<bool, std::string> pdb::PDBBufferManagerFrontEnd::handleFreezeSizeRequest(pdb::Handle<pdb::BufFreezeSizeRequest> &request, std::shared_ptr<T> &sendUsingMe) {

  // if this is an anonymous page the set is a null ptr
  PDBSetPtr set = nullptr;

  // if this is not an anonymous page create a set
  if(!request->isAnonymous) {
    set = make_shared<PDBSet>(*request->databaseName, *request->setName);
  }

  // create the page key
  auto key = std::make_pair(set, request->pageNumber);

  PDBPageHandle handle;
  bool res;
  {
    // lock the thing
    unique_lock<mutex> lck(m);

    // find if the thing exists
    auto it = sentPages.find(key);

    // did we find it?
    res = it != sentPages.end();

    // grab the page handle
    handle = it->second;
  }

  // if we did find it freeze it
  if(res) {

    // freeze it!
    handle->freezeSize(request->freezeSize);
  }

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response
  Handle<BufFreezeRequestResult> response = makeObject<BufFreezeRequestResult>(res);

  // sends result to requester
  std::string errMsg;
  res = sendUsingMe->sendObject(response, errMsg) && res;

  // return
  return make_pair(res, errMsg);
}

template <class T>
std::pair<bool, std::string> pdb::PDBBufferManagerFrontEnd::handlePinPageRequest(pdb::Handle<pdb::BufPinPageRequest> &request, std::shared_ptr<T> &sendUsingMe) {
  // if this is an anonymous page the set is a null ptr
  PDBSetPtr set = nullptr;

  // if this is not an anonymous page create a set
  if(!request->isAnonymous) {
    set = make_shared<PDBSet>(*request->databaseName, *request->setName);
  }

  bool res;
  PDBPageHandle handle;
  {
    // lock the thing
    unique_lock<mutex> lck(m);

    // create the page key
    auto key = std::make_pair(set, request->pageNumber);

    // find if the thing exists
    auto it = sentPages.find(key);

    // did we find it?
    res = it != sentPages.end();

    // grab a handle
    handle = it->second;
  }

  // if we did find it, if so pin it
  if(res) {

    // pin it
    handle->repin();
  }

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response
  Handle<BufPinPageResult> response = makeObject<BufPinPageResult>((uint64_t) handle->page->bytes - (uint64_t) sharedMemory.memory, res);

  // sends result to requester
  std::string errMsg;
  res = sendUsingMe->sendObject(response, errMsg) && res;

  // return
  return make_pair(res, errMsg);
}

template <class T>
std::pair<bool, std::string> pdb::PDBBufferManagerFrontEnd::handleUnpinPageRequest(pdb::Handle<pdb::BufUnpinPageRequest> &request, std::shared_ptr<T> &sendUsingMe) {

  // if this is an anonymous page the set is a null ptr
  PDBSetPtr set = nullptr;

  // if this is not an anonymous page create a set
  if(!request->isAnonymous) {
    set = make_shared<PDBSet>(*request->databaseName, *request->setName);
  }

  bool res;
  PDBPageHandle handle;
  {
    // lock the thing
    unique_lock<mutex> lck(m);

    // create the page key
    auto key = std::make_pair(set, request->pageNumber);

    // find if the thing exists
    auto it = sentPages.find(key);

    // did we find it?
    res = it != sentPages.end();

    // grab a handle
    handle = it->second;
  }

  // if we did find it, if so unpin it
  if(res) {

    // update the dirty bit
    if(request->isDirty) {
      handle->setDirty();
    }

    // unpin it
    handle->unpin();
  }

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response
  Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(res, res ? std::string("") : std::string("Could not find the page to unpin page!"));

  // sends result to requester
  std::string errMsg;
  res = sendUsingMe->sendObject(response, errMsg) && res;

  // return
  return make_pair(res, errMsg);
}

template<class T>
bool pdb::PDBBufferManagerFrontEnd::handleForwardPage(pdb::PDBPageHandle &page, shared_ptr<T> &communicator, std::string &error) {

  // this method must never be called with an unpinned page
  assert (page->isPinned());

  // init the forwarding process
  initForwarding(page);

  /// 1. Sent the request

  auto offset = (uint64_t) page->page->bytes - (uint64_t) sharedMemory.memory;
  auto pageNum = page->whichPage();
  auto isAnon = page->page->isAnon;
  auto sizeFrozen = page->page->sizeFrozen;
  auto startPos = page->page->getLocation().startPos;
  auto numBytes = page->page->getLocation().numBytes;

  std::string setName;
  std::string dbName;

  if(!isAnon) {

    // set the database name and set name
    dbName = page->getSet()->getDBName();
    setName = page->getSet()->getSetName();
  }

  // make an allocation block
  const UseTemporaryAllocationBlock tempBlock{1024};

  // forward the page
  Handle<pdb::BufForwardPageRequest> objectToSend = pdb::makeObject<BufForwardPageRequest>(offset, pageNum, isAnon, sizeFrozen, startPos, numBytes, setName, dbName);

  // log the forward
  logForward(objectToSend);

  // send the object
  std::string errMsg;
  if (!communicator->sendObject(objectToSend, errMsg)) {

    // yeah something happened
    logger->error(errMsg);
    logger->error("Not able to send request to server.\n");

    // we are done here we do not recover from this error
    return false;
  }

  // log that the object is sent
  logger->info("Forward request sent.");

  /// 2. Get the response

  // get the response and process it
  size_t objectSize = communicator->getSizeOfNextObject();

  // check if we did get a response
  if (objectSize == 0) {

    // ok we did not that sucks log what happened
    logger->error("We did not get a response.\n");

    // we are done here we do not recover from this error
    return false;
  }

  // allocate the memory
  std::unique_ptr<char[]> memory(new char[objectSize]);
  if (memory == nullptr) {

    errMsg = "FATAL ERROR in heapRequest: Can't allocate memory";
    logger->error(errMsg);

    /// TODO this needs to be an exception or something
    // this is a fatal error we should not be running out of memory
    exit(-1);
  }

  // grab the result
  bool success;
  Handle<SimpleRequestResult> result = communicator->template getNextObject<SimpleRequestResult> (memory.get(), success, errMsg);
  if (!success) {

    // log the error
    logger->error(errMsg);
    logger->error("not able to get next object over the wire.\n");

    // we are done here we do not recover from this error
    return false;
  }

  // finish forwarding
  finishForwarding(page);

  // we are done here
  return success;
}

template <class T>
bool pdb::PDBBufferManagerFrontEnd::sendPageToBackend(pdb::PDBPageHandle page, std::shared_ptr<T> &sendUsingMe, std::string &error) {

  // figure out the page parameters
  auto offset = (uint64_t) page->page->bytes - (uint64_t) sharedMemory.memory;
  auto pageNumber = page->whichPage();
  auto isAnonymous = page->page->isAnonymous();
  auto sizeFrozen = page->page->sizeIsFrozen();
  auto startPos = page->page->location.startPos;
  auto numBytes = page->page->location.numBytes;

  // make an allocation block
  const UseTemporaryAllocationBlock tempBlock{1024};

  std::string setName = isAnonymous ? "" : page->getSet()->getSetName();
  std::string dbName = isAnonymous ? "" : page->getSet()->getDBName();

  // create the object
  Handle<pdb::BufGetPageResult> objectToSend = pdb::makeObject<BufGetPageResult>(offset, pageNumber, isAnonymous, sizeFrozen, startPos, numBytes, setName, dbName);

  {
    // lock so we can mark the page as sent
    unique_lock<mutex> lck(m);

    // mark that we have sent the page, store a handle so that we keep the reference count
    sentPages[std::make_pair(page->getSet(), pageNumber)] = page;
  }

  // send the thing
  bool res = sendUsingMe->sendObject(objectToSend, error);

  // did we fail?
  if(!res) {

    // if we failed do a cleanup
    unique_lock<mutex> lck(m);

    // erase the stuff that failed
    sentPages.erase(std::make_pair(page->getSet(), pageNumber));
  }

  // return the result
  return res;
}


#endif //PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_H
