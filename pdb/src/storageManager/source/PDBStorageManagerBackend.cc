//
// Created by dimitrije on 2/11/19.
//

#include <PDBBufferManagerBackEnd.h>
#include <SharedEmployee.h>
#include "PDBStorageManagerBackend.h"
#include "HeapRequestHandler.h"
#include "StoStoreOnPageRequest.h"
#include "StoSetStatsRequest.h"
#include "StoSetStatsResult.h"
#include <boost/filesystem/path.hpp>
#include <PDBSetPageSet.h>

void pdb::PDBStorageManagerBackend::init() {

  // init the class
  logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "logs").string(), "PDBStorageManagerBackend.log");
}

void pdb::PDBStorageManagerBackend::registerHandlers(PDBServer &forMe) {

  forMe.registerHandler(
      StoStoreOnPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoStoreOnPageRequest>>(
          [&](Handle<pdb::StoStoreOnPageRequest> request, PDBCommunicatorPtr sendUsingMe) {
            return handleStoreOnPage(request, sendUsingMe);
      }));
}

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManagerBackend::getPageSet(const std::string &db, const std::string &set) {

  auto conf = getConfiguration();

  auto numPages = RequestFactory::heapRequest<StoSetStatsRequest, StoSetStatsResult, std::pair<bool, uint64_t>>(
      logger, conf->managerPort, conf->managerAddress, std::make_pair<bool, uint64_t>(false, 0), 1024,
      [&](Handle<StoSetStatsResult> result) {

        // do we have a result if not return false
        if (result == nullptr) {

          logger->error("Failed to get the number of pages for a page set created for the following PDBSet : (" + db + "," + set + ")");
          return std::make_pair<bool, uint64_t>(false, 0);
        }

        // did we succeed
        if (!result->success) {

          logger->error("Failed to get the number of pages for a page set created for the following PDBSet : (" + db + "," + set + ")");
          return std::make_pair<bool, uint64_t>(false, 0);
        }

        // we succeeded
        return std::make_pair(result->success, result->size);
      });

  // if we failed return a null ptr
  if(!numPages.first) {
    return nullptr;
  }

  return std::make_shared<pdb::PDBSetPageSet>(db, set, numPages.second, getFunctionalityPtr<PDBBufferManagerInterface>());
}