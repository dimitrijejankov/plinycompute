#include <utility>

//
// Created by dimitrije on 3/5/19.
//

#include <PDBSetPageSet.h>

#include "PDBSetPageSet.h"

pdb::PDBSetPageSet::PDBSetPageSet(const std::string &db,
                                  const std::string &set,
                                  vector<uint64_t> &pages,
                                  pdb::PDBBufferManagerInterfacePtr bufferManager) : curPage(0), pages(pages), bufferManager(std::move(bufferManager)) {
  // make the pdb set
  this->set = make_shared<PDBSet>(db, set);
}

pdb::PDBPageHandle pdb::PDBSetPageSet::getNextPage(size_t workerID) {

  // figure out the current page
  uint64_t pageNum = curPage++;

  // if we are out of pages return null
  if(pageNum >= pages.size()) {
    return nullptr;
  }

  // return the page
  return bufferManager->getPage(set, pages[pageNum]);
}

pdb::PDBPageHandle pdb::PDBSetPageSet::getNewPage() {

  // just throw since we won't need this for some time
  throw runtime_error("Adding a new page to a page set not implemented.");
}

size_t pdb::PDBSetPageSet::getNumPages() {
  return pages.size();
}

void pdb::PDBSetPageSet::resetPageSet() {

  // reset the page counter
  curPage = 0;
}
