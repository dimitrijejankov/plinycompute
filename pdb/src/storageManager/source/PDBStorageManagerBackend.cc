//
// Created by dimitrije on 2/11/19.
//

#include <PDBBufferManagerBackEnd.h>
#include <SharedEmployee.h>
#include "HeapRequestHandler.h"
#include "StoStoreOnPageRequest.h"
#include "StoSetStatsRequest.h"
#include "StoSetStatsResult.h"
#include <boost/filesystem/path.hpp>
#include <PDBSetPageSet.h>
#include <PDBStorageManagerBackend.h>
#include <StoStartWritingToSetRequest.h>
#include <StoStartWritingToSetResult.h>

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

pdb::PDBSetPageSetPtr pdb::PDBStorageManagerBackend::createPageSetFromPDBSet(const std::string &db, const std::string &set,
                                                                             const std::pair<uint64_t, std::string> &pageSetID) {


  // get the configuration
  auto conf = this->getConfiguration();

  /// 1. Contact the frontend and to get the number of pages

  auto numPages = RequestFactory::heapRequest<StoSetStatsRequest, StoSetStatsResult, std::pair<bool, uint64_t>>(
      logger, conf->port, conf->address, std::make_pair<bool, uint64_t>(false, 0), 1024,
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
        return std::make_pair(result->success, result->numPages);
      }, db, set);

  // if we failed return a null ptr
  if(!numPages.first) {
    return nullptr;
  }

  /// 2. Check if we already have the thing if we do return it

  std::unique_lock<std::mutex> lck(pageSetMutex);

  // try to find the page if it exists return it
  auto it = pageSets.find(pageSetID);
  if(it != pageSets.end()) {
    return std::dynamic_pointer_cast<PDBSetPageSet>(it->second);
  }

  /// 3. We don't have it so create it

  // store the page set
  auto pageSet = std::make_shared<pdb::PDBSetPageSet>(db, set, numPages.second, getFunctionalityPtr<PDBBufferManagerInterface>());
  pageSets[pageSetID] = pageSet;

  // return it
  return pageSet;
}

pdb::PDBAnonymousPageSetPtr pdb::PDBStorageManagerBackend::createAnonymousPageSet(const std::pair<uint64_t, std::string> &pageSetID) {

  /// 1. Check if we already have the thing if we do return it

  std::unique_lock<std::mutex> lck(pageSetMutex);

  // try to find the page if it exists return it
  auto it = pageSets.find(pageSetID);
  if(it != pageSets.end()) {
    return std::dynamic_pointer_cast<PDBAnonymousPageSet>(it->second);
  }

  /// 2. We don't have it so create it

  // store the page set
  auto pageSet = std::make_shared<pdb::PDBAnonymousPageSet>(getFunctionalityPtr<PDBBufferManagerInterface>());
  pageSets[pageSetID] = pageSet;

  // return it
  return pageSet;
}

pdb::PDBAbstractPageSetPtr pdb::PDBStorageManagerBackend::getPageSet(const std::pair<uint64_t, std::string> &pageSetID) {

  // try to find the page if it exists return it
  auto it = pageSets.find(pageSetID);
  if(it != pageSets.end()) {
    return std::dynamic_pointer_cast<PDBAbstractPageSet>(it->second);
  }

  // return null since we don't have it
  return nullptr;
}

bool pdb::PDBStorageManagerBackend::materializePageSet(pdb::PDBAbstractPageSetPtr pageSet, const std::pair<std::string, std::string> &set) {

  // number of pages that we need to write
  auto numPages = pageSet->getNumPages();

  // request from the frontend to start writing the pages
  auto response = RequestFactory::heapRequest<StoStartWritingToSetRequest, StoStartWritingToSetResult, std::pair<bool, uint64_t>>(
      this->logger, getConfiguration()->port, getConfiguration()->address, std::make_pair(false, 0), 1024,
      [&](Handle<StoStartWritingToSetResult> result) {

        // if the result is something else null we got a response
        if (result == nullptr) {
          return std::make_pair<bool, uint64_t>(false, 0);
        }

        //
        return std::make_pair(result->success, result->startPage);
      },
      numPages, set.first, set.second);

  // did we fail?
  if(!response.first) {
    return false;
  }

  // buffer manager
  auto bufferManager = getFunctionalityPtr<PDBBufferManagerInterface>();
  auto setIdentifier = std::make_shared<PDBSet>(set.first, set.second);

  // control variables
  PDBPageHandle page;
  uint64_t currPage = response.second;

  // ok we did not let's copy the stuff
  while ((page = pageSet->getNextPage(0)) != nullptr) {

    // repin the page
    page->repin();

    // get the set page
    auto setPage = bufferManager->getPage(setIdentifier, currPage++);

    // get the size of the page
    auto pageSize = page->getSize();

    // copy the stuff
    memcpy(setPage->getBytes(), page->getBytes(), pageSize);

    // freeze the size
    setPage->freezeSize(pageSize);
  }

  return true;
}
