//
// Created by dimitrije on 3/8/19.
//

#include "PDBAnonymousPageSet.h"

pdb::PDBAnonymousPageSet::PDBAnonymousPageSet(const pdb::PDBBufferManagerInterfacePtr &bufferManager) : bufferManager(bufferManager), isDone(false) {
  curPage = pages.begin();
}

pdb::PDBPageHandle pdb::PDBAnonymousPageSet::getNextPage(size_t workerID) {

  // are we done if so return null
  if(isDone) {
    return nullptr;
  }

  // lock so we can mess with the data structure
  std::unique_lock<std::mutex> lck(m);

  // grab the current page
  auto page = curPage->second;

  // go to the next page
  curPage++;

  // if done mark as done
  isDone = curPage == pages.end();

  // return the page
  return page;
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
