#pragma once

#include <unordered_map>
#include <utility>

namespace pdb {

// setup the ptr type
class PDBLabeledPageSet;
using PDBLabeledPageSetPtr = std::shared_ptr<PDBLabeledPageSet>;

/**
 * The view of a set for a particular label
 * every page created by this page set is assigned the same label
 */
class PDBLabeledPageSetView : public PDBAbstractPageSet {
public:

  PDBLabeledPageSetView(int32_t label,
                        PDBAbstractPageSetPtr pageSet,
                        unordered_map<int32_t, int32_t> *labels,
                        mutex *m) : label(label),
                                    pageSet(std::move(pageSet)),
                                    labels(labels),
                                    m(m) {}

  /**
   * Gets a new page from the page set and returns
   * @return - the new page
   */
  PDBPageHandle getNewPage() override {

    // get a new page
    auto page = pageSet->getNewPage();

    // lock the mutex
    std::lock_guard<std::mutex> lck(*m);

    // set the label
    (*labels)[page->whichPage()] = label;

    return page;
  }

  /**
   * Remove the page from the page set
   * @param pageHandle - the handle
   */
  void removePage(PDBPageHandle pageHandle) override {

    // remove the page
    pageSet->removePage(pageHandle);

    // lock the mutex
    std::lock_guard<std::mutex> lck(*m);

    // remove the page
    (*labels).erase(pageHandle->whichPage());
  }

  /**
   * @details Not supported
   */
  PDBPageHandle getNextPage(size_t workerID) override {
    throw runtime_error("You can not iterate over the PDBLabeledPageSetView");
  }

  /**
   * @details Not supported
   */
  size_t getNumPages() override {
    throw runtime_error("You can not get the number of pages in PDBLabeledPageSetView use the PDBLabeledPageSet for that.");
  }

  /**
   * @details Not supported
   */
  void resetPageSet() override {
    throw runtime_error("You can not reset a PDBLabeledPageSetView");
  }

  /**
   * Return the maximum page size of the page set
   * @return returns max page size
   */
  size_t getMaxPageSize() override {
    return pageSet->getMaxPageSize();
  }

 private:

  // the label
  int32_t label;

  // the page set
  PDBAbstractPageSetPtr pageSet;

  // the labels
  std::unordered_map<int32_t, int32_t> *labels;

  // the mutex
  std::mutex *m;
};

/**
 * This class acts as a wrapper for a particular page set.
 * @see getLabel - returns the label assigned to a particular page
 * @see getLabeledView - this creates a view of the page set where each call to new page is going to
 * assign a label to each getNewPage call.
 */
class PDBLabeledPageSet : public PDBAbstractPageSet {
public:

  // the constructor
  PDBLabeledPageSet() = default;

  // the constructor
  explicit PDBLabeledPageSet(const PDBAbstractPageSetPtr &pageSet) : pageSet(pageSet) {}

  /**
   * Return the next page in the page set
   * @param workerID - the id of the worker
   * @return the next page
   */
  PDBPageHandle getNextPage(size_t workerID) override {
    return pageSet->getNextPage(workerID);
  }

  /**
   * @details this is not supported by the labeled set use a view for that
   * @throws always throws an exception
   */
  PDBPageHandle getNewPage() override {
    throw runtime_error("You can not get a new page from a PDBLabeledPageSet, use a view to do that.");
  }

  /**
   * Removes the page from the page set
   * @param pageHandle - the handle
   */
  void removePage(PDBPageHandle pageHandle) override {

    // remove the page
    pageSet->removePage(pageHandle);

    // lock the mutex
    std::lock_guard<std::mutex> lck(m);

    // remove the page
    labels.erase(pageHandle->whichPage());
  }

  /**
   * Returns the number of pages in the contained page set
   * @return the number of pages
   */
  size_t getNumPages() override {
    return pageSet->getNumPages();
  }

  /**
   * Reset the contained page set
   */
  void resetPageSet() override {
    pageSet->resetPageSet();
  }

  /**
   * Returns the maximum page size
   * @return the maximum page size
   */
  size_t getMaxPageSize() override {
    return pageSet->getMaxPageSize();
  }

  /**
   * Returns the label of the page if it has one, otherwise it returns -1
   * @param page - the page
   * @return return the label
   */
  int32_t getLabel(const PDBPageHandle &page) {

    // lock the mutex
    std::lock_guard<std::mutex> lck(m);

    // try to find the label for the page
    auto it = labels.find(page->whichPage());
    if(it != labels.end()) {
      return it->second;
    }

    // we have not found it
    return -1;
  }

  /**
   * Sets the label of a page
   * @param page - the page
   * @param label - the label
   */
  PDBAbstractPageSetPtr getLabeledView(int32_t label) {
    return std::make_shared<PDBLabeledPageSetView>(label, pageSet, &labels, &m);
  }

  // the labels for each page
  std::unordered_map<int32_t, int32_t> labels;

  // the page set
  PDBAbstractPageSetPtr pageSet;

  // the mutex
  std::mutex m;
};



}
