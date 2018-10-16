#include <PDBStorageManagerFrontEnd.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <PagedRequestHandler.h>
#include <StoGetPageRequest.h>
#include <PDBStorageManagerBackEnd.h>
#include <StoGetAnonymousPageRequest.h>
#include <PagedRequest.h>
#include <StoGetPageResult.h>
#include <SimpleRequestResult.h>
#include <StoReturnPageRequest.h>
#include <StoReturnAnonPageRequest.h>

void pdb::PDBStorageManagerFrontEnd::init() {

  // init the logger
  //logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "PDBStorageManagerFrontend.log").string());
  logger = make_shared<pdb::PDBLogger>("PDBStorageManagerFrontend.log");
}

void pdb::PDBStorageManagerFrontEnd::registerHandlers(pdb::PDBServer &forMe) {
  forMe.registerHandler(StoGetPageRequest_TYPEID,
      make_shared<pdb::PagedRequestHandler<StoGetPageRequest>>(
      [&](Handle<StoGetPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // grab the page
        auto page = getPage(std::make_shared<PDBSet>(request->setName, request->dbName), request->pageNumber);

        // send the page to the backend
        std::string error;
        bool res = sendPageToBackend(page, sendUsingMe, error);

        return make_pair(res, error);
      }));

  forMe.registerHandler(StoGetAnonymousPageRequest_TYPEID,
      make_shared<pdb::PagedRequestHandler<StoGetAnonymousPageRequest>>([&](Handle<StoGetAnonymousPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // grab an anonymous page
        auto page = getPage(request->size);

        // send the page to the backend
        std::string error;
        bool res = sendPageToBackend(page, sendUsingMe, error);

        return make_pair(res, error);
      }));

  forMe.registerHandler(StoReturnPageRequest_TYPEID,
      make_shared<pdb::PagedRequestHandler<StoReturnPageRequest>>([&](Handle<StoReturnPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // create the page key
        auto key = std::make_pair(std::make_shared<PDBSet>(request->databaseName, request->setName), request->pageNumber);

        // try to remove it, if we manage to do this res will be true
        bool res = this->sentPages.erase(key) != 0;

        // create an allocation block to hold the response
        auto bufferPage = getPage(1024);
        const UseTemporaryAllocationBlock tempBlock{bufferPage->getBytes(), 1024};

        // create the response
        Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(res, res ? std::string("") : std::string("Could not find the page to remove!"));

        // sends result to requester
        std::string errMsg;
        res = sendUsingMe->sendObject(response, errMsg) && res;

        // return
        return make_pair(res, errMsg);
      }));

  forMe.registerHandler(StoReturnAnonPageRequest_TYPEID,
        make_shared<pdb::PagedRequestHandler<StoReturnAnonPageRequest>>([&](Handle<StoReturnAnonPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

          // create the page key
          auto key = std::make_pair((PDBSetPtr) nullptr, request->pageNumber);

          // try to remove it, if we manage to do this res will be true
          bool res = this->sentPages.erase(key) != 0;

          // create an allocation block to hold the response
          auto bufferPage = getPage(1024);
          const UseTemporaryAllocationBlock tempBlock{bufferPage->getBytes(), 1024};

          // create the response
          Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(res, res ? std::string("") : std::string("Could not find the page to remove!"));

          // sends result to requester
          std::string errMsg;
          res = sendUsingMe->sendObject(response, errMsg) && res;

          // return
          return make_pair(res, errMsg);
        }));
}

bool pdb::PDBStorageManagerFrontEnd::sendPageToBackend(pdb::PDBPageHandle page, pdb::PDBCommunicatorPtr sendUsingMe, std::string &error) {

  // figure out the page parameters
  auto offset = (uint64_t) page->page->bytes - (uint64_t) sharedMemory.memory;
  auto pageNumber = page->whichPage();
  auto isAnonymous = page->page->isAnonymous();
  auto sizeFrozen = page->page->sizeIsFrozen();
  auto startPos = page->page->location.startPos;
  auto numBytes = page->page->location.numBytes;

  // grab the buffer page
  auto bufferPage = getPage(1024);
  const UseTemporaryAllocationBlock tempBlock{bufferPage->getBytes(), 1024};

  std::string setName = isAnonymous ? "" : page->getSet()->getSetName();
  std::string dbName = isAnonymous ? "" : page->getSet()->getDBName();

  // create the object
  Handle<StoGetPageResult> objectToSend = pdb::makeObject<StoGetPageResult>(offset, pageNumber, isAnonymous, sizeFrozen, startPos, numBytes, setName, dbName);

  // send the thing
  bool res = sendUsingMe->sendObject(objectToSend, error);

  // did we succeed
  if(res) {

    // mark that we have sent the page, store a handle so that we keep the reference count
    sentPages[std::make_pair(page->getSet(), pageNumber)] = page;
  }

  // return the result
  return res;
}

pdb::PDBStorageManagerInterfacePtr pdb::PDBStorageManagerFrontEnd::getBackEnd() {

  // init the backend storage manager with the shared memory
  return std::make_shared<PDBStorageManagerBackEnd>(sharedMemory);
}
