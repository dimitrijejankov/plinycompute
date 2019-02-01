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
#include <StoFreezeSizeRequest.h>
#include <StoPinPageRequest.h>
#include <StoUnpinPageRequest.h>
#include <StoPinPageResult.h>
#include <HeapRequestHandler.h>


pdb::PDBStorageManagerFrontEnd::PDBStorageManagerFrontEnd(std::string tempFileIn, size_t pageSizeIn, size_t numPagesIn, std::string metaFile, std::string storageLocIn) {

  // initialize the buffer manager
  initialize(std::move(tempFileIn), pageSizeIn, numPagesIn, std::move(metaFile), std::move(storageLocIn));
}

void pdb::PDBStorageManagerFrontEnd::init() {

  // init the logger
  //logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "PDBStorageManagerFrontend.log").string());
  logger = make_shared<pdb::PDBLogger>("PDBStorageManagerFrontend.log");
}

void pdb::PDBStorageManagerFrontEnd::registerHandlers(pdb::PDBServer &forMe) {
  forMe.registerHandler(StoGetPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<StoGetPageRequest>>(
          [&](Handle<StoGetPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handleGetPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(StoGetAnonymousPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<StoGetAnonymousPageRequest>>(
          [&](Handle<StoGetAnonymousPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handleGetAnonymousPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(StoReturnPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<StoReturnPageRequest>>(
          [&](Handle<StoReturnPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handleReturnPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(StoReturnAnonPageRequest_TYPEID, make_shared<pdb::HeapRequestHandler<StoReturnAnonPageRequest>>(
          [&](Handle<StoReturnAnonPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handleReturnAnonPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(StoFreezeSizeRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<StoFreezeSizeRequest>>(
          [&](Handle<StoFreezeSizeRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handleFreezeSizeRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(StoPinPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<StoPinPageRequest>>([&](Handle<StoPinPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handlePinPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(StoUnpinPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<StoUnpinPageRequest>>([&](Handle<StoUnpinPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

        // call the method to handle it
        return handleUnpinPageRequest(request, sendUsingMe);
      }));
}

pdb::PDBStorageManagerInterfacePtr pdb::PDBStorageManagerFrontEnd::getBackEnd() {

  // init the backend storage manager with the shared memory
  return std::make_shared<PDBStorageManagerBackEnd<RequestFactory>>(sharedMemory);
}



