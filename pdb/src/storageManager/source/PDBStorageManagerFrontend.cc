//
// Created by dimitrije on 2/9/19.
//

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
#include <StoMaterializePageSetRequest.h>
#include <StoStartFeedingPageSetRequest.h>

namespace fs = boost::filesystem;

pdb::PDBStorageManagerFrontend::~PDBStorageManagerFrontend() {

  // open the output file
  std::ofstream ofs;
  ofs.open((boost::filesystem::path(getConfiguration()->rootDirectory) / "storage.pdb").string(),
           ios::binary | std::ofstream::out | std::ofstream::trunc);

  unsigned long numSets = pageStats.size();
  ofs.write((char *) &numSets, sizeof(unsigned long));

  // serialize the stuff
  for (auto &it : pageStats) {

    // write the database name
    unsigned long size = it.first->getDBName().size();
    ofs.write((char *) &size, sizeof(unsigned long));
    ofs.write(it.first->getDBName().c_str(), size);

    // write the set name
    size = it.first->getSetName().size();
    ofs.write((char *) &size, sizeof(unsigned long));
    ofs.write(it.first->getSetName().c_str(), size);

    // write the page stats
    ofs.write(reinterpret_cast<char *>(&it.second), sizeof(it.second));
  }

  ofs.close();
}

void pdb::PDBStorageManagerFrontend::init() {

  // init the class
  logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "logs").string(),
                                       "PDBStorageManagerFrontend.log");

  // do we have the metadata for the storage
  if (fs::exists(boost::filesystem::path(getConfiguration()->rootDirectory) / "storage.pdb")) {

    // open if stream
    std::ifstream ifs;
    ifs.open((boost::filesystem::path(getConfiguration()->rootDirectory) / "storage.pdb").string(),
             ios::binary | std::ifstream::in);

    size_t numSets;
    ifs.read((char *) &numSets, sizeof(unsigned long));

    for (int i = 0; i < numSets; ++i) {

      // read the database name
      unsigned long size;
      ifs.read((char *) &size, sizeof(unsigned long));
      std::unique_ptr<char[]> setBuffer(new char[size]);
      ifs.read(setBuffer.get(), size);
      std::string dbName(setBuffer.get(), size);

      // read the set name
      ifs.read((char *) &size, sizeof(unsigned long));
      std::unique_ptr<char[]> dbBuffer(new char[size]);
      ifs.read(dbBuffer.get(), size);
      std::string setName(dbBuffer.get(), size);

      // read the number of pages
      PDBStorageSetStats pageStat{};
      ifs.read(reinterpret_cast<char *>(&pageStat), sizeof(pageStat));

      // store the set info
      auto set = std::make_shared<PDBSet>(dbName, setName);
      this->pageStats[set] = pageStat;
    }

    // close
    ifs.close();
  }
}

void pdb::PDBStorageManagerFrontend::registerHandlers(PDBServer &forMe) {

  forMe.registerHandler(
      StoGetPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoGetPageRequest>>([&](Handle<pdb::StoGetPageRequest> request,
                                                                       PDBCommunicatorPtr sendUsingMe) {
        // handle the get page request
        return handleGetPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoDispatchData_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoDispatchData>>(
          [&](Handle<pdb::StoDispatchData> request, PDBCommunicatorPtr sendUsingMe) {

            return handleDispatchedData<PDBCommunicator, RequestFactory>(request, sendUsingMe);
          }));

  forMe.registerHandler(
      StoGetSetPagesRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoGetSetPagesRequest>>([&](pdb::Handle<pdb::StoGetSetPagesRequest> request, PDBCommunicatorPtr sendUsingMe) {
        return handleGetSetPages<PDBCommunicator, RequestFactory>(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoMaterializePageSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoMaterializePageSetRequest>>([&](pdb::Handle<pdb::StoMaterializePageSetRequest> request, PDBCommunicatorPtr sendUsingMe) {
        return handleMaterializeSet<PDBCommunicator, RequestFactory>(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoRemovePageSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoRemovePageSetRequest>>([&](Handle<pdb::StoRemovePageSetRequest> request, PDBCommunicatorPtr sendUsingMe) {
        return handleRemovePageSet(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoStartFeedingPageSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoStartFeedingPageSetRequest>>([&](Handle<pdb::StoStartFeedingPageSetRequest> request, PDBCommunicatorPtr sendUsingMe) {
        return handleStartFeedingPageSetRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoClearSetRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoClearSetRequest>>([&](Handle<pdb::StoClearSetRequest> request, PDBCommunicatorPtr sendUsingMe) {
        return handleClearSetRequest(request, sendUsingMe);
  }));

}

bool pdb::PDBStorageManagerFrontend::isPageBeingWrittenTo(const pdb::PDBSetPtr &set, uint64_t pageNum) {

  // try to find it
  auto it = pagesBeingWrittenTo.find(set);

  // do we even have it here
  if(it == pagesBeingWrittenTo.end()) {
    return false;
  }

  // check if is in the set of free pages
  return it->second.find(pageNum) != it->second.end();}

bool pdb::PDBStorageManagerFrontend::isPageFree(const pdb::PDBSetPtr &set, uint64_t pageNum) {

  // try to find it
  auto it = freeSkippedPages.find(set);

  // do we even have it here
  if(it == freeSkippedPages.end()) {
    return false;
  }

  // check if is in the set of free pages
  return it->second.find(pageNum) != it->second.end();
}

bool pdb::PDBStorageManagerFrontend::pageExists(const pdb::PDBSetPtr &set, uint64_t pageNum) {

  // try to find the page
  auto it = this->pageStats.find(set);

  // if it exists and is smaller or equal to the last page then it exists
  return it != this->pageStats.end() && pageNum <= it->second.lastPage;
}

std::pair<bool, uint64_t> pdb::PDBStorageManagerFrontend::getValidPage(const pdb::PDBSetPtr &set, uint64_t pageNum) {

  // lock the stuff
  unique_lock<std::mutex> lck(pageMutex);

  // try to find the page
  auto it = this->pageStats.find(set);

  // do we even have stats about this, if not finish
  if(it == this->pageStats.end()) {
    return make_pair(false, 0);
  }

  // do we even have this page
  while(pageNum <= it->second.lastPage) {

    // check if the page is valid
    if(pageExists(set, pageNum) && !isPageBeingWrittenTo(set, pageNum) && !isPageFree(set, pageNum)) {
      return make_pair(true, pageNum);
    }

    // if not try to go to the next one
    pageNum++;
  }

  // finish
  return make_pair(false, 0);
}

uint64_t pdb::PDBStorageManagerFrontend::getNextFreePage(const pdb::PDBSetPtr &set) {

  // see if we have a free page already
  auto pages = freeSkippedPages.find(set);
  if(!pages->second.empty()) {

    // get the page number
    auto page = *pages->second.begin();

    // remove the thing
    pages->second.erase(pages->second.begin());

    // return the page
    return page;
  }

  // try to find the set
  auto it = pageStats.find(set);

  // do we even have a record for this set
  uint64_t pageNum;
  if(it == pageStats.end()) {

    // set the page to zero since this is the first page
    pageStats[set].lastPage = 0;
    pageNum = 0;

    // set the page size
    pageStats[set].size = 0;
  }
  else {

    // increment the last page
    pageNum = ++it->second.lastPage;
  }

  return pageNum;
}

void pdb::PDBStorageManagerFrontend::incrementSetSize(const pdb::PDBSetPtr &set, uint64_t uncompressedSize) {

  // try to find the set
  auto it = pageStats.find(set);

  // increment the set size on this node
  it->second.size += uncompressedSize;
}

void pdb::PDBStorageManagerFrontend::freeSetPage(const pdb::PDBSetPtr &set, uint64_t pageNum) {

  // insert the page into the free list
  freeSkippedPages[set].insert(pageNum);
}

void pdb::PDBStorageManagerFrontend::startWritingToPage(const pdb::PDBSetPtr &set, uint64_t pageNum) {
  // mark the page as being written to
  pagesBeingWrittenTo[set].insert(pageNum);
}

void pdb::PDBStorageManagerFrontend::endWritingToPage(const pdb::PDBSetPtr &set, uint64_t pageNum) {
  // unmark the page as being written to
  pagesBeingWrittenTo[set].erase(pageNum);
}

bool pdb::PDBStorageManagerFrontend::handleDispatchFailure(const PDBSetPtr &set, uint64_t pageNum, uint64_t size, PDBCommunicatorPtr communicator) {

  // where we put the error
  std::string error;

  {
    // lock the stuff
    unique_lock<std::mutex> lck(pageMutex);

    // finish writing to the set
    endWritingToPage(set, pageNum);

    // return the page to the free list
    freeSetPage(set, pageNum);

    // decrement back the set size
    decrementSetSize(set, size);
  }

  // create an allocation block to hold the response
  Handle<SimpleRequestResult> failResponse = makeObject<SimpleRequestResult>(false, error);

  // sends result to requester
  return communicator->sendObject(failResponse, error);
}

void pdb::PDBStorageManagerFrontend::decrementSetSize(const pdb::PDBSetPtr &set, uint64_t uncompressedSize) {

  // try to find the set
  auto it = pageStats.find(set);

  // increment the set size on this node
  it->second.size -= uncompressedSize;
}

std::shared_ptr<pdb::PDBStorageSetStats> pdb::PDBStorageManagerFrontend::getSetStats(const PDBSetPtr &set) {

  // try to find the set
  auto it = pageStats.find(set);

  // if we have it return it
  if(it != pageStats.end()){
    return std::make_shared<pdb::PDBStorageSetStats>(it->second);
  }

  // return null
  return nullptr;
}
