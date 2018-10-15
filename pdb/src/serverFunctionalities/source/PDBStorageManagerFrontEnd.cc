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

void pdb::PDBStorageManagerFrontEnd::init() {

  // init the logger
  logger = make_shared<pdb::PDBLogger>(boost::filesystem::path(getConfiguration()->rootDirectory) / "PDBStorageManagerFrontend.log");
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
      make_shared<pdb::PagedRequestHandler<StoGetAnonymousPageRequest>>([&](Handle<StoGetAnonymousPageRequest> request,
                                                                            PDBCommunicatorPtr sendUsingMe) {

        // grab an anonymous page
        auto page = getPage(request->size);

        // send the page to the backend
        std::string error;
        bool res = sendPageToBackend(page, sendUsingMe, error);

        return make_pair(res, error);
      }));
}

bool pdb::PDBStorageManagerFrontEnd::sendPageToBackend(pdb::PDBPageHandle page, pdb::PDBCommunicatorPtr sendUsingMe, std::string &error) {

  auto config = getConfiguration();

  return pagedRequest<StoGetPageResult, SimpleRequestResult, bool>(
      config->address, config->port, getFunctionalityPtr<PDBStorageManagerInterface>(), logger, false, 1024,
      [&](Handle<SimpleRequestResult> result) {
        if (result != nullptr) {

          if (!result->getRes().first) {

            // log the error
            error = "Error sending page to backend : " + result->getRes().second;
            logger->error(error);

            // return false
            return false;
          }
        }
        return true;
      }, 0, false, false, 0, false, false, 0, 0, "", "");
}

pdb::PDBStorageManagerInterfacePtr pdb::PDBStorageManagerFrontEnd::getBackEnd() {

  // init the backend storage manager with the shared memory
  return std::make_shared<PDBStorageManagerBackEnd>(sharedMemory);
}
