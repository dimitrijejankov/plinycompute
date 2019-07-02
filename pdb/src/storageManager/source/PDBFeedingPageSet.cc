#include <utility>


#include <assert.h>
#include <PDBFeedingPageSet.h>

#include "PDBFeedingPageSet.h"

pdb::PDBFeedingPageInfo::PDBFeedingPageInfo(pdb::PDBPageHandle page, uint64_t numUsers, uint64_t timesServed)
    : page(std::move(page)), numUsers(numUsers), timesServed(timesServed) {}

pdb::PDBFeedingPageSet::PDBFeedingPageSet(uint64_t numReaders, uint64_t numFeeders) : nextPageForWorker(numReaders), numReaders(numReaders), numFeeders(numFeeders), numFinishedFeeders(0), nextPage(0) {}

pdb::PDBPageHandle pdb::PDBFeedingPageSet::getNewPage() {
  throw runtime_error("One can only add the pages to the feeding page set.");
}

size_t pdb::PDBFeedingPageSet::getNumPages() {
  return pages.size();
}

pdb::PDBPageHandle pdb::PDBFeedingPageSet::getNextPage(size_t workerID) {

  // lock pages structure
  unique_lock<std::mutex> lck(m);

  // go to the next page
  auto page = nextPageForWorker[workerID]++;

  // wait to have a page
  cv.wait(lck, [&]{ return numFinishedFeeders == numFeeders || nextPage > page; });

  // if we are done here (all feeders have finished and we served the last page) return null
  if(numFinishedFeeders == numFeeders && nextPage == page) {
    return nullptr;
  }

  // if the policy is that we remove them after they are being served then do just that
  if(page != 0 && usagePolicy == PDBFeedingPageSetUsagePolicy::REMOVE_AFTER_USED) {

    // find the last page
    auto lastPage = pages.find(page - 1);

    // this must always be true otherwise something is wrong
    assert(lastPage != pages.end());

    // decrement the number of users since we are not using this page anymore
    lastPage->second.numUsers--;

    // check if somebody is using the page
    if(lastPage->second.numUsers == 0) {

      // unpin the page
      lastPage->second.page->unpin();

      // have we served this page to all readers? if so remove it
      if(lastPage->second.timesServed == numReaders) {
        pages.erase(lastPage);
      }
    }
  }

  // find the page
  auto it = pages.find(page);

  // this must always be true otherwise something is wrong
  assert(it != pages.end());

  // we only need to update the stats iff we plan on removing them after they are used
  if(usagePolicy == PDBFeedingPageSetUsagePolicy::REMOVE_AFTER_USED) {

    // update the stats for the served page
    it->second.numUsers++;
    it->second.timesServed++;
  }

  // repin and return the page
  it->second.page->repin();
  return it->second.page;
}

void pdb::PDBFeedingPageSet::feedPage(const PDBPageHandle &page) {

  // lock pages structure
  unique_lock<std::mutex> lck(m);

  if(numFinishedFeeders == numFeeders) {
    throw runtime_error("Trying to feed pages, when all feeders have done feeding.");
  }

  // insert the page
  pages.insert(std::make_pair(nextPage++, PDBFeedingPageInfo(page, 0, 0)));

  // unlock and notify that we inserted a page
  lck.unlock();
  cv.notify_all();
}

void pdb::PDBFeedingPageSet::finishFeeding() {

  // lock pages structure
  unique_lock<std::mutex> lck(m);

  // increment the number of finished feeders
  numFinishedFeeders++;

  // notify that we are done feeding
  lck.unlock();
  cv.notify_all();
}

void pdb::PDBFeedingPageSet::resetPageSet() {

  // lock pages structure
  unique_lock<std::mutex> lck(m);

  // reset the next pages
  for_each(nextPageForWorker.begin(), nextPageForWorker.end(), [&](auto &page) {page = 0;});
}

void pdb::PDBFeedingPageSet::setUsagePolicy(pdb::PDBFeedingPageSetUsagePolicy policy) {

  // lock the feeding page set
  unique_lock<std::mutex> lck(m);

  // set the policy
  this->usagePolicy = policy;
}