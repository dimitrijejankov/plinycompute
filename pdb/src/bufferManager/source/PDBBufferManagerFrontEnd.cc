#include <PDBBufferManagerFrontEnd.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <PagedRequestHandler.h>
#include <BufGetPageRequest.h>
#include <PDBBufferManagerBackEnd.h>
#include <BufGetAnonymousPageRequest.h>
#include <PagedRequest.h>
#include <BufGetPageResult.h>
#include <SimpleRequestResult.h>
#include <BufReturnPageRequest.h>
#include <BufReturnAnonPageRequest.h>
#include <BufFreezeSizeRequest.h>
#include <BufPinPageRequest.h>
#include <BufUnpinPageRequest.h>
#include <BufPinPageResult.h>
#include <HeapRequestHandler.h>
#include <BufForwardPageRequest.h>

pdb::PDBBufferManagerFrontEnd::PDBBufferManagerFrontEnd(std::string tempFileIn, size_t pageSizeIn, size_t numPagesIn, std::string metaFile, std::string storageLocIn) {

  // initialize the buffer manager
  initialize(std::move(tempFileIn), pageSizeIn, numPagesIn, std::move(metaFile), std::move(storageLocIn));
}

void pdb::PDBBufferManagerFrontEnd::init() {

  // init the logger
  //logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "PDBStorageManagerFrontend.log").string());
  logger = make_shared<pdb::PDBLogger>("PDBStorageManagerFrontend.log");
}

bool pdb::PDBBufferManagerFrontEnd::forwardPage(pdb::PDBPageHandle &page, pdb::PDBCommunicatorPtr &communicator, std::string &error) {

  // handle the page forwarding request
  return handleForwardPage(page, communicator, error);
}

void pdb::PDBBufferManagerFrontEnd::finishForwarding(pdb::PDBPageHandle &page)  {

    // lock so we can mark the page as sent
    unique_lock<mutex> lck(this->m);

    // mark that we have finished forwarding
    this->forwarding.erase(make_pair(page->getSet(), page->page->whichPage()));
}

void pdb::PDBBufferManagerFrontEnd::initForwarding(pdb::PDBPageHandle &page) {

    // lock so we can mark the page as sent
    unique_lock<mutex> lck(this->m);

    // make the key
    pair<PDBSetPtr, long> key = std::make_pair(page->getSet(), page->whichPage());

    // wait if there is a forward of the page is happening
    cv.wait(lck, [&] { return !(forwarding.find(key) != forwarding.end()); });

    // mark the page as sent
    this->sentPages[make_pair(page->getSet(), page->page->whichPage())] = page;

    // mark that we are forwarding the page
    this->forwarding.insert(make_pair(page->getSet(), page->page->whichPage()));
}

void pdb::PDBBufferManagerFrontEnd::registerHandlers(pdb::PDBServer &forMe) {
  forMe.registerHandler(BufGetPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufGetPageRequest>>(
          [&](Handle<BufGetPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handleGetPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(BufGetAnonymousPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufGetAnonymousPageRequest>>(
          [&](Handle<BufGetAnonymousPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handleGetAnonymousPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(BufReturnPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufReturnPageRequest>>(
          [&](Handle<BufReturnPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handleReturnPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(BufReturnAnonPageRequest_TYPEID, make_shared<pdb::HeapRequestHandler<BufReturnAnonPageRequest>>(
          [&](Handle<BufReturnAnonPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handleReturnAnonPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(BufFreezeSizeRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufFreezeSizeRequest>>(
          [&](Handle<BufFreezeSizeRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handleFreezeSizeRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(BufPinPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufPinPageRequest>>([&](Handle<BufPinPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handlePinPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(BufUnpinPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufUnpinPageRequest>>([&](Handle<BufUnpinPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handleUnpinPageRequest(request, sendUsingMe);
      }));
}

pdb::PDBBufferManagerInterfacePtr pdb::PDBBufferManagerFrontEnd::getBackEnd() {

  // init the backend storage manager with the shared memory
  return std::make_shared<PDBBufferManagerBackEnd<RequestFactory>>(sharedMemory);
}



