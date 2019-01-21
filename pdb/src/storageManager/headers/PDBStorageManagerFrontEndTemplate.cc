//
// Created by dimitrije on 1/21/19.
//

#ifndef PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_CC
#define PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_CC

#include <SimpleRequestResult.h>
#include <StoPinPageResult.h>

template <class T>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontEnd::handleGetPageRequest(pdb::Handle<pdb::StoGetPageRequest> &request, std::shared_ptr<T> &sendUsingMe) {

  // grab the page
  auto page = this->getPage(make_shared<pdb::PDBSet>(request->setName, request->dbName), request->pageNumber);

  // send the page to the backend
  string error;
  bool res = this->sendPageToBackend(page, sendUsingMe, error);

  return make_pair(res, error);
}

template <class T>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontEnd::handleGetAnonymousPageRequest(pdb::Handle<pdb::StoGetAnonymousPageRequest> &request, std::shared_ptr<T> &sendUsingMe) {

  // grab an anonymous page
  auto page = getPage(request->size);

  // send the page to the backend
  std::string error;
  bool res = sendPageToBackend(page, sendUsingMe, error);

  return make_pair(res, error);
}

template <class T>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontEnd::handleReturnPageRequest(pdb::Handle<pdb::StoReturnPageRequest> &request, std::shared_ptr<T> &sendUsingMe) {

  // create the page key
  auto key = std::make_pair(std::make_shared<PDBSet>(request->setName, request->databaseName), request->pageNumber);

  // try to remove it, if we manage to do this res will be true
  bool res = this->sentPages.erase(key) != 0;

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
std::pair<bool, std::string> pdb::PDBStorageManagerFrontEnd::handleReturnAnonPageRequest(pdb::Handle<pdb::StoReturnAnonPageRequest> &request, std::shared_ptr<T> &sendUsingMe) {

  // create the page key
  auto key = std::make_pair((PDBSetPtr) nullptr, request->pageNumber);

  // try to remove it, if we manage to do this res will be true
  bool res = this->sentPages.erase(key) != 0;

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
std::pair<bool, std::string> pdb::PDBStorageManagerFrontEnd::handleFreezeSizeRequest(pdb::Handle<pdb::StoFreezeSizeRequest> &request, std::shared_ptr<T> &sendUsingMe) {

  // if this is an anonymous page the set is a null ptr
  PDBSetPtr set = nullptr;

  // if this is not an anonymous page create a set
  if(!request->isAnonymous) {
    set = make_shared<PDBSet>(*request->setName, *request->databaseName);
  }

  // create the page key
  auto key = std::make_pair(set, request->pageNumber);

  // find if the thing exists
  auto it = sentPages.find(key);

  // did we find it?
  bool res = it != sentPages.end();

  // if we did find it freeze it
  if(res) {

    // freeze it!
    it->second->freezeSize(request->freezeSize);
  }

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response
  Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(res, res ? std::string("") : std::string("Could not find the page to freeze!"));

  // sends result to requester
  std::string errMsg;
  res = sendUsingMe->sendObject(response, errMsg) && res;

  // return
  return make_pair(res, errMsg);
}

template <class T>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontEnd::handlePinPageRequest(pdb::Handle<pdb::StoPinPageRequest> &request, std::shared_ptr<T> &sendUsingMe) {
  // if this is an anonymous page the set is a null ptr
  PDBSetPtr set = nullptr;

  // if this is not an anonymous page create a set
  if(!request->isAnonymous) {
    set = make_shared<PDBSet>(*request->setName, *request->databaseName);
  }

  // create the page key
  auto key = std::make_pair(set, request->pageNumber);

  // find if the thing exists
  auto it = sentPages.find(key);

  // did we find it?
  bool res = it != sentPages.end();

  // if we did find it, if so pin it
  if(res) {

    // pin it
    it->second->repin();
  }

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response
  Handle<StoPinPageResult> response = makeObject<StoPinPageResult>((uint64_t) it->second->page->bytes - (uint64_t) sharedMemory.memory, res);

  // sends result to requester
  std::string errMsg;
  res = sendUsingMe->sendObject(response, errMsg) && res;

  // return
  return make_pair(res, errMsg);
}

template <class T>
std::pair<bool, std::string> pdb::PDBStorageManagerFrontEnd::handleUnpinPageRequest(pdb::Handle<pdb::StoUnpinPageRequest> &request, std::shared_ptr<T> &sendUsingMe) {

  // if this is an anonymous page the set is a null ptr
  PDBSetPtr set = nullptr;

  // if this is not an anonymous page create a set
  if(!request->isAnonymous) {
    set = make_shared<PDBSet>(*request->setName, *request->databaseName);
  }

  // create the page key
  auto key = std::make_pair(set, request->pageNumber);

  // find if the thing exists
  auto it = sentPages.find(key);

  // did we find it?
  bool res = it != sentPages.end();

  // if we did find it, if so unpin it
  if(res) {

    // unpin it
    it->second->unpin();
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

#endif //PDB_PDBSTORAGEMANAGERFRONTENDTEMPLATE_H
