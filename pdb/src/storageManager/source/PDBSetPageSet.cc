//
// Created by dimitrije on 3/5/19.
//

#include <PDBSetPageSet.h>

#include "PDBSetPageSet.h"

pdb::PDBSetPageSet::PDBSetPageSet(const std::string &db,
                                  const std::string &set,
                                  size_t numPages,
                                  pdb::PDBBufferManagerInterfacePtr bufferManager) : db(db), set(set), curPage(0), lastPage(numPages) {

  // make the pdb set
  PDBSetPtr pdbSet = make_shared<PDBSet>(db, set);

  // allocate the pages
  pages.reserve(numPages);

  // grab the page handles
  for(uint64_t i = 0; i < numPages; ++i) {

    // grab a page
    auto page = bufferManager->getPage(pdbSet, i);

    // unpin it
    page->unpin();

    pages.emplace_back(page);
  }
}

pdb::PDBPageHandle pdb::PDBSetPageSet::getNextPage(size_t workerID) {

  // figure out the current page
  uint64_t pageNum = curPage++;

  // if we are out of pages return null
  if(pageNum >= pages.size()) {
    return nullptr;
  }

  // grab the page and repin it...
  auto page = pages[pageNum];
  page->repin();

  // return the page
  return page;
}

pdb::PDBPageHandle pdb::PDBSetPageSet::getNewPage() {

  // figure out the next page
  uint64_t pageNum = lastPage++;

  // grab a page
  return bufferManager->getPage(make_shared<PDBSet>(db, set), pageNum);
}
