//
// Created by dimitrije on 3/8/19.
//

#include "PDBAnonymousPageSet.h"

pdb::PDBAnonymousPageSet::PDBAnonymousPageSet(const pdb::PDBBufferManagerInterfacePtr &bufferManager) : bufferManager(bufferManager) {}

pdb::PDBPageHandle pdb::PDBAnonymousPageSet::getNextPage(size_t workerID) {

  // lock so we can mess with the data structure
  std::unique_lock<std::mutex> lck(m);

  // if we don't have pages return null
  if (pages.empty()) {
    return nullptr;
  }

  // in the case that we are doing a sequential access we are simply going to treat each worker as the same worker with the index 0
  workerID = accessPattern == PDBAnonymousPageSetAccessPattern::CONCURRENT ? workerID : 0;

  // do we need to initialize the iterator to the start
  if(nextPageForWorker.find(workerID) == nextPageForWorker.end()) {

    // set the current page to start
    nextPageForWorker[workerID] = pages.begin();
  }

  // grab the current page
  auto &curPage = nextPageForWorker[workerID];

  // is this the last page for this worker if so end
  if(curPage == pages.end()) {
    return nullptr;
  }

  // grab the page handle from it
  auto pageHandle = curPage->second;

  // go to the next page
  curPage++;

  // return the page
  return pageHandle;
}

pdb::PDBPageHandle pdb::PDBAnonymousPageSet::getNewPage() {

  // grab an anonymous page
  auto page = bufferManager->getPage();

  // lock the pages struct
  {
    std::unique_lock<std::mutex> lck(m);

    // add the page
    pages[page->whichPage()] = page;
  }

  return page;
}

void pdb::PDBAnonymousPageSet::removePage(pdb::PDBPageHandle pageHandle) {

  // lock the pages struct
  std::unique_lock<std::mutex> lck(m);

  // remove the page
  pages.erase(pageHandle->whichPage());
}

size_t pdb::PDBAnonymousPageSet::getNumPages() {

  // lock the pages struct
  std::unique_lock<std::mutex> lck(m);

  // return the size
  return pages.size();
}

void pdb::PDBAnonymousPageSet::resetPageSet() {

  // lock the pages struct
  std::unique_lock<std::mutex> lck(m);

  // reset the current pages
  nextPageForWorker.clear();
}

void pdb::PDBAnonymousPageSet::setAccessOrder(PDBAnonymousPageSetAccessPattern pattern) {

  // lock the pages struct
  std::unique_lock<std::mutex> lck(m);

  // set the pattern
  accessPattern = pattern;

  // reset the current pages
  nextPageForWorker.clear();
}