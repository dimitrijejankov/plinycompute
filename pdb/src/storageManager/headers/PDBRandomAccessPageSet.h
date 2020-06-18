#pragma once

#include <PDBAbstractPageSet.h>
#include <PDBBufferManagerInterface.h>

#include <utility>

namespace pdb {

class PDBRandomAccessPageSet : public PDBAbstractPageSet {

public:

  PDBRandomAccessPageSet(PDBBufferManagerInterfacePtr bufferManager) : bufferManager(std::move(bufferManager)) {}

  PDBPageHandle getNextPage(size_t label) override {

    // lock stuff
    unique_lock lck(m);

    // get the index
    auto index = labelToIndex[label];

    // if the index is less
    if(index < pages.size()) {
      return pages[index];
    }

    // set the label
    labelToIndex[label] = ++index;

    // we don't have a page
    return nullptr;
  }

  PDBPageHandle getNewPage() override {

    // lock stuff
    unique_lock lck(m);

    // insert a new page
    auto page = bufferManager->getPage();
    pages.emplace_back(page);

    // return it
    return std::move(page);
  }

  std::tuple<size_t, PDBPageHandle> getNextPageWithIndex() {

    // lock stuff
    unique_lock lck(m);

    // insert a new page
    auto page = bufferManager->getPage();
    pages.emplace_back(page);

    // get the last index
    auto index = pages.size() - 1;

    // return the tuple
    return std::make_tuple(index, page);
  }

  void removePage(PDBPageHandle pageHandle) override {
    throw runtime_error("You can not remove a page from a random access page set!");
  }

  size_t getNumPages() override {

    // lock stuff
    unique_lock lck(m);

    // return the number of pages
    return pages.size();
  }

  void resetPageSet() override {

    // lock stuff
    unique_lock lck(m);

    // clear the labels
    labelToIndex.clear();
  }

  size_t getMaxPageSize() override {
    return bufferManager->getMaxPageSize();
  }

  PDBPageHandle &operator[](int index) {

    // lock stuff
    unique_lock lck(m);

    // return the page
    return pages[index];
  }

  void repinAll() {

    // lock stuff
    unique_lock lck(m);

    // repin all the pages
    for(const auto &page : pages){
      page->repin();
    }
  }

private:

  /**
   * Mutex to sync the pages map
   */
  std::mutex m;

  /**
   * This maps the worker to it's index
   */
  std::unordered_map<size_t, uint64_t> labelToIndex;

  /**
   * The pages in this page set
   */
  std::vector<PDBPageHandle> pages;

  /**
   * the buffer manager to get the pages
   */
  PDBBufferManagerInterfacePtr bufferManager;
};

using PDBRandomAccessPageSetPtr = std::shared_ptr<PDBRandomAccessPageSet>;

}