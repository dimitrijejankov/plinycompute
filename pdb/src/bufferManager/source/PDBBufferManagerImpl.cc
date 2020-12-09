/****************************************************
** COPYRIGHT 2016, Chris Jermaine, Rice University **
**                                                 **
** The MyDB Database System, COMP 530              **
** Note that this file contains SOLUTION CODE for  **
** A1.  You should not be looking at this file     **
** unless you have completed A1!                   **
****************************************************/


#ifndef STORAGE_MGR_C
#define STORAGE_MGR_C

#include "PDBPage.h"
#include "PDBBufferManagerFileWriter.h"
#include "PDBBufferManagerImpl.h"

#include <fcntl.h>
#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <stdio.h>
#include <sys/mman.h>
#include <cstring>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <PDBBufferManagerImpl.h>

namespace pdb {

namespace fs = boost::filesystem;

PDBBufferManagerImpl::PDBBufferManagerImpl(const pdb::NodeConfigPtr& config) {

  // create the root directory
  fs::path dataPath(config->rootDirectory);
  dataPath.append("/data");

  // check if we have the metadata
  if (fs::exists(dataPath / "metadata.pdb")) {

    // log what happened
    std::cout << "Using an existing storage!\n";

    // ok we found a previous storage init it with that
    initialize((dataPath / "metadata.pdb").string());

    // we are done here
    return;
  }

  // no previous storage found start a new m
  std::cout << "Starting a new storage!\n";

  // create the data directory
  if (!fs::exists(dataPath) && !fs::create_directories(dataPath)) {
    std::cout << "Failed to create the data directory!\n";
  }

  // grab the memory size and he page size
  auto memorySize = config->sharedMemSize * 1024 * 1024;
  auto pageSize = config->pageSize;

  // just a quick sanity check
  if (pageSize == 0 || memorySize == 0) {
    throw std::runtime_error("The memory size or the page size can not be 0");
  }

  // figure out the number of pages we have available
  auto numPages = memorySize / pageSize;

  // init the new manager
  initialize((dataPath / "tempFile___.tmp").string(),
             pageSize,
             numPages,
             (dataPath / "metadata.pdb").string(),
             dataPath.string());
}

size_t PDBBufferManagerImpl::getMaxPageSize() {

  if (!initialized) {
    cerr << "Can't call getMaxPageSize () without initializing the storage manager\n";
    exit(1);
  }

  return sharedMemory.pageSize;
}

PDBPagePtr PDBBufferManagerImpl::getPageForObject(void *objectAddress) {
        void* startSharedMem = (void*)sharedMemory.memory;
        void* endSharedMem = (void*)((char*)sharedMemory.memory + sharedMemory.numPages * sharedMemory.pageSize);
        void *memLoc = (char *) sharedMemory.memory
                   + ((((char *)objectAddress - (char *) sharedMemory.memory) / sharedMemory.pageSize) * sharedMemory.pageSize);

        if (objectAddress < startSharedMem || objectAddress > endSharedMem){
            std::cerr << objectAddress << " is not in the range of shared memory. range start: " << startSharedMem << " end: " << endSharedMem << "\n";
            exit(-1);
        }
        for (const auto& miniPage: constituentPages[memLoc]){
            char* start = (char*)miniPage->getBytes();
            char* end = start + miniPage->getSize();
            if (objectAddress >= start && objectAddress < end) {
                return miniPage;
            }
        }
        std::cerr << "GetPageForObject: cannot find page for this object \n";
        std::cerr << "GetPageForObject: object address is: " << objectAddress << '\n';
        exit(-1);
}


void PDBBufferManagerImpl::cleanup() {

  if (!initialized)
    return;

  // loop through all of the pages currently in existence, and write back each of them
  for (auto &a : allPages) {

    // if the page is not dirty, do not do anything
    PDBPagePtr &me = a.second;
    if (!me->isDirty())
      continue;

    pair<PDBSetPtr, long> whichPage = make_pair(me->getSet(), me->whichPage());

    // if we don't know where to write it, figure it out
    if (pageLocations.count(whichPage) == 0) {
      me->getLocation().startPos = endOfFiles[me->getSet()];
      pair<PDBSetPtr, size_t> myIndex = make_pair(me->getSet(), me->whichPage());
      pageLocations[myIndex] = me->getLocation();
      endOfFiles[me->getSet()] += (MIN_PAGE_SIZE << me->getLocation().numBytes);
    }

    ssize_t write_bytes = pwrite(fds[me->getSet()],
                                 me->getBytes(),
                                 MIN_PAGE_SIZE << me->getLocation().numBytes,
                                 me->getLocation().startPos);
    if (write_bytes == -1) {
      std::cerr << "error in PDBBufferManager destructor when writing data to disk with errno: " << strerror(errno)
                << std::endl;
      exit(1);
    }
  }

  // free the memory
  delete[] static_cast<char*>(sharedMemory.memory);

  // write the new meta data
  remove(metaDataFile.c_str());
  PDBBufferManagerFileWriter myMetaFile(metaDataFile);

  // now, write out the meta-data
  myMetaFile.putUnsignedLong("pageSize", sharedMemory.pageSize);
  myMetaFile.putLong("logOfPageSize", logOfPageSize);
  myMetaFile.putUnsignedLong("numPages", sharedMemory.numPages);
  myMetaFile.putString("tempFile", tempFile);
  myMetaFile.putString("metaDataFile", metaDataFile);
  myMetaFile.putString("storageLoc", storageLoc);

  // get info on each of the sets
  vector<string> setNames;
  vector<string> dbNames;
  vector<string> endPos;
  for (auto &a : endOfFiles) {
    setNames.push_back(a.first->getSetName());
    dbNames.push_back(a.first->getDBName());
    endPos.push_back(to_string(a.second));
  }

  myMetaFile.putStringList("setNames", setNames);
  myMetaFile.putStringList("dbNames", dbNames);
  myMetaFile.putStringList("endPosForEachFile", endPos);

  // now, for each set, write the list of page positions
  map<string, vector<string>> allFiles;
  for (auto &a : pageLocations) {
    string key = a.first.first->getSetName() + "." + a.first.first->getDBName();
    string value = to_string(a.first.second) + "." + to_string(a.second.startPos) + "." + to_string(a.second.numBytes);
    if (allFiles.count(key) == 0) {
      vector<string> temp;
      temp.push_back(value);
      allFiles[key] = temp;
    } else {
      allFiles[key].push_back(value);
    }
  }

  // and now write out all of the page positions
  for (auto &a : allFiles) {
    myMetaFile.putStringList(a.first, a.second);
  }

  myMetaFile.save();
}

void PDBBufferManagerImpl::initialize(const std::string& meta) {

  // mark the manager as initialized
  initialized = true;

  // first, get the basic info and initialize everything
  PDBBufferManagerFileWriter myMetaFile(meta);

  // we use these to grab metadata
  uint64_t utemp;
  int64_t temp;

  myMetaFile.getUnsignedLong("pageSize", utemp);
  sharedMemory.pageSize = utemp;
  myMetaFile.getLong("logOfPageSize", temp);
  logOfPageSize = temp;
  myMetaFile.getUnsignedLong("numPages", utemp);
  sharedMemory.numPages = utemp;
  myMetaFile.getString("tempFile", tempFile);
  myMetaFile.getString("storageLoc", storageLoc);

  initialize(tempFile, sharedMemory.pageSize, sharedMemory.numPages, meta, storageLoc);

  // now, get everything that we need for the end positions
  vector<string> setNames;
  vector<string> dbNames;
  vector<string> endPos;

  myMetaFile.getStringList("setNames", setNames);
  myMetaFile.getStringList("dbNames", dbNames);
  myMetaFile.getStringList("endPosForEachFile", endPos);

  // and set up the end positions
  for (int i = 0; i < setNames.size(); i++) {
    PDBSetPtr mySet = make_shared<PDBSet>(dbNames[i], setNames[i]);
    size_t end = stoul(endPos[i]);
    endOfFiles[mySet] = end;
  }

  // now, we need to get the file mappings
  for (auto &a : endOfFiles) {
    vector<string> mappings;
    myMetaFile.getStringList(a.first->getSetName() + "." + a.first->getDBName(), mappings);
    for (auto &b : mappings) {

      // the format of the string in the list is pageNum.startPos.len
      stringstream tokenizer(b);
      string pageNum, startPos, len;
      getline(tokenizer, pageNum, '.');
      getline(tokenizer, startPos, '.');
      getline(tokenizer, len, '.');

      size_t page = stoul(pageNum);
      PDBPageInfo tempInfo;
      tempInfo.startPos = stoul(startPos);
      tempInfo.numBytes = stoul(len);
      pageLocations[make_pair(a.first, page)] = tempInfo;
    }
  }
}

void PDBBufferManagerImpl::initialize(const std::string& tempFileIn, size_t pageSizeIn, size_t numPagesIn,
                                      std::string metaFile, std::string storageLocIn) {
  initialized = true;
  storageLoc = std::move(storageLocIn);
  sharedMemory.pageSize = pageSizeIn;
  tempFile = tempFileIn;
  sharedMemory.numPages = numPagesIn;
  metaDataFile = std::move(metaFile);

  tempFileFD = open(tempFileIn.c_str(), O_CREAT | O_RDWR, 0666);
  if (tempFileFD == -1) {
    std::cerr << "Fail to open the temp file at " << tempFile << std::endl;
    exit(1);
  }

  // there are no currently available positions
  logOfPageSize = -1;
  size_t curSize;
  for (curSize = MIN_PAGE_SIZE; curSize <= sharedMemory.pageSize; curSize *= 2) {
    vector<int64_t> temp;
    vector<void *> tempAgain;
    availablePositions.push_back(temp);
    emptyMiniPages.push_back(tempAgain);
    isCreatingSpace.push_back(false);
    logOfPageSize++;
  }

  // but the last used position is zero
  lastTempPos = 0;

  // as is the last time tick
  lastTimeTick = 1;

  if (curSize != sharedMemory.pageSize * 2) {
    std::cerr << "Error: the page size must be a power of two.\n";
    exit(1);
  }

  // now, allocate the RAM
  char *mapped;
  mapped = new char[sharedMemory.pageSize * sharedMemory.numPages];

  // make sure that it actually worked
  if (mapped == nullptr) {
    exit(1);
  }

  sharedMemory.memory = mapped;

  // and create a bunch of pages
  for (int i = 0; i < sharedMemory.numPages; i++) {

    // figure out the address
    void *address = mapped + (sharedMemory.pageSize * i);

    // store the address
    emptyFullPages.push_back(address);
  }

}

void PDBBufferManagerImpl::clearSet(const PDBSetPtr &set) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lock(m);

  // figure out all the pages we need to remove
  bool found = false;
  std::vector<std::pair<size_t, PDBPagePtr>> pagesToRemove;
  for(auto & allPage : allPages) {

    // if the set does not match got to the next
    if(*allPage.first.first != *set) {

      // if we previously found a page then we can safely
      // finish searching for pages since they are sorted
      if (found) {
        break;
      }

      // go to the next page and continue
      continue;
    }

    // did we find it...
    found = true;

    // store the page
    pagesToRemove.emplace_back(allPage.first.second, allPage.second);
  }

  // remove the pages from all the pages
  // the pages are sorted, so there is a optimization that once we find a page then every next page has to
  // be from the target set otherwise we know that there are not pages from the target set left
  for(auto &page : pagesToRemove) {

    // wait while are doing something with the page
    pagesCV.wait(lock, [&] { return !(page.second->status == PDB_PAGE_LOADING ||
                                             page.second->status == PDB_PAGE_UNLOADING ||
                                             page.second->status == PDB_PAGE_FREEZING ||
                                             page.second->status == PDB_PAGE_MOVING); });

    // find the parent page of this page
    void *memLoc = (char *) sharedMemory.memory + ((((char *) page.second->getBytes() - (char *) sharedMemory.memory) / sharedMemory.pageSize) * sharedMemory.pageSize);
    auto &siblings = constituentPages[memLoc];

    // if this is not found
    if (page.second->status == PDB_PAGE_NOT_LOADED) {

      // remove the page from all pages
      auto it = allPages.find({set, page.first});
      allPages.erase(it);
      continue;
    }

    // decrease the number of pinned pages if needed
    if(page.second->isPinned()) {

      // decrement the number of pinned pages
      numPinned[memLoc]--;
    }

    // remove the page from the constituentPages
    auto pagePos = std::find(siblings.begin(), siblings.end(), page.second);
    siblings.erase(pagePos);

    // if we just uppinned the whole page we need to add it back into circulation
    if(numPinned[memLoc] == 0) {

      // if by removing the page the constituent pages are empty we can return the whole page back
      if(siblings.empty()) {

        // first we need to remove all the mini pages that this page is split into and are not used
        auto &unused = unusedMiniPages[memLoc];
        auto &emptyPages = emptyMiniPages[unused.second];

        // go through each unused mini page and remove it!
        for (auto const &kt : unused.first) {
          auto const jt = std::find(emptyPages.begin(), emptyPages.end(), kt);
          emptyPages.erase(jt);
        }

        // clear the unused pages
        unused.first.clear();

        // add back the full page
        emptyFullPages.push_back(memLoc);
      }
      else {

        // otherwise reinsert it as a free page and add it to the LRU
        unusedMiniPages[memLoc].first.emplace_back(page.second->bytes);
        emptyMiniPages[page.second->location.numBytes].emplace_back(page.second->bytes);

        // add it to the LRU
        numPinned[memLoc] = -lastTimeTick;

        // add the physical page to the LRU
        lastUsed.insert(make_pair(memLoc, lastTimeTick));

        // increment the time tick
        lastTimeTick++;
      }
    } else {

      // ok it is not pinned but we still need to free the minipage
      unusedMiniPages[memLoc].first.emplace_back(page.second->bytes);
      emptyMiniPages[page.second->location.numBytes].emplace_back(page.second->bytes);
    }

    // remove the page from all pages
    auto it = allPages.find({set, page.first});
    allPages.erase(it);
  }

  // remove all page locations for that set
  for(auto it = pageLocations.begin(); it != pageLocations.end(); ) {

    // check if the set matches
    if (*it->first.first == *set) {

      // remove the location since the set matches
      pageLocations.erase(it++);
      continue;
    }

    // go to next
    ++it;
  }

  // close the file if open
  auto fd = fds.find(set);
  if(fd != fds.end()) {
    close(fd->second);
    fds.erase(fd);
  }

  // remove the end of files
  endOfFiles.erase(set);

  // log the clear set
  logClearSet(set);
}

void PDBBufferManagerImpl::movePage(const PDBSetPtr& whichSet, uint64_t pageNumber, const PDBPagePtr &anonymousPage) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lock(m);

  // check if the page already exists at that location
  auto it = pageLocations.find(std::make_pair(whichSet, pageNumber));
  if(it != pageLocations.end()) {
    cerr << "Something went wrong, when moving the set page we are moving to must be empty.";
    exit(-1);
  }

  // check if everything is alright
  if(!anonymousPage->isPinned()) {
    cerr << "Something went wrong, when moving the page the anonymous page needs to be pinned.";
    exit(-1);
  }

  // check if we have the file for this set...
  checkIfOpen(whichSet);

  // rewire the set
  anonymousPage->setSet(whichSet);

  // return the location if necessary
  PDBPageInfo temp = anonymousPage->getLocation();
  if (temp.startPos != -1) {
    availablePositions[temp.numBytes].push_back(temp.startPos);
  }

  // set the new page number
  anonymousPage->pageNum = pageNumber;

  // identifier of the page
  pair<PDBSetPtr, size_t> whichPage = std::make_pair(whichSet, pageNumber);

  // mark that this is now a set page
  anonymousPage->setAnonymous(false);

  // if the size is frozen add it to the end of the file
  if(anonymousPage->sizeFrozen) {

    // if we don't know where to write it, figure it out
    anonymousPage->getLocation().startPos = endOfFiles[anonymousPage->getSet()];
    pair<PDBSetPtr, size_t> myIndex = make_pair(anonymousPage->getSet(), anonymousPage->whichPage());
    pageLocations[whichPage] = anonymousPage->getLocation();
    endOfFiles[anonymousPage->getSet()] += (MIN_PAGE_SIZE << anonymousPage->getLocation().numBytes);
  }

  // insert the page
  allPages[whichPage] = anonymousPage;
}

void PDBBufferManagerImpl::registerMiniPage(const PDBPagePtr& registerMe) {

  // first, compute the page this guy is on
  void *whichPage = (char *) sharedMemory.memory
      + ((((char *) registerMe->getBytes() - (char *) sharedMemory.memory) / sharedMemory.pageSize)
          * sharedMemory.pageSize);

  // now, add him to the list of constituent pages
  constituentPages[whichPage].push_back(registerMe);

  // this guy is now pinned
  pinParent(registerMe);
}

void PDBBufferManagerImpl::freeAnonymousPage(PDBPagePtr me) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lock(m);

  // is this removal still valid if it is not we do nothing
  if (!isRemovalStillValid(me)) {
    return;
  }

  // recycle his location if it had a location assigned to it
  PDBPageInfo temp = me->getLocation();
  if (temp.startPos != -1) {
    availablePositions[temp.numBytes].push_back(temp.startPos);
  }

  // return the anonymous page number
  freeAnonPageNumbers.emplace_back(me->whichPage());

  // if this guy as no associated memory, get outta here
  if (me->getBytes() == nullptr) {

    // log free anonymous page set
    logFreeAnonymousPage(me->whichPage());

    // get out of here
    return;
  }

  // now, remove him from the set of constituent pages
  void *parent = (char *) sharedMemory.memory + ((((char *) me->getBytes() - (char *) sharedMemory.memory) / sharedMemory.pageSize) * sharedMemory.pageSize);

  // remove this page from the mini pages
  auto &miniPages = constituentPages[parent];

  // remove the mini page
  auto pagePos = std::find(miniPages.begin(), miniPages.end(), me);
  miniPages.erase(pagePos);

  // reduce the number of pinned pages if pinned
  numPinned[parent] -= me->pinned ? 1 : 0;

  // if we don't have any mini pages on the parent page, we can kill the page
  if(miniPages.empty()) {

    // if the page is already in the LRU remove it from the LRU
    if (numPinned[parent] < 0) {
      pair<void *, unsigned> id = make_pair(parent, -numPinned[parent]);
      lastUsed.erase(id);
    }

    // remove the unused pages
    auto &unused = unusedMiniPages[parent];
    auto &emptyPages = emptyMiniPages[unused.second];

    // go through each unused mini page and remove it!
    for (auto const &kt : unused.first) {
      auto const jt = std::find(emptyPages.begin(), emptyPages.end(), kt);
      emptyPages.erase(jt);
    }

    // clear the unused pages
    unused.first.clear();

    // add back the empty full page
    emptyFullPages.push_back(parent);

    // log free anonymous page set
    logFreeAnonymousPage(me->whichPage());

    // finish this
    return;
  }

  // if the number of pinned minipages is now zero, put into the LRU structure
  if (numPinned[parent] == 0) {

    // add it to the LRU
    numPinned[parent] = -lastTimeTick;

    // add the physical page to the LRU
    lastUsed.insert(make_pair(parent, lastTimeTick));

    // increment the time tick
    lastTimeTick++;
  }

  // add back the unused mini pages
  unusedMiniPages[parent].first.emplace_back(me->getBytes());
  emptyMiniPages[unusedMiniPages[parent].second].emplace_back(me->getBytes());

  // log free anonymous page set
  logFreeAnonymousPage(me->whichPage());

  // notify that we have created space
  spaceCV.notify_all();
}

void PDBBufferManagerImpl::downToZeroReferences(PDBPagePtr me) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lock(m);

  // is this removal still valid if it is not we do nothing
  if (!isRemovalStillValid(me)) {
    return;
  }

  // first, see whether we are actually buffered in RAM
  if (me->getBytes() == nullptr) {

    // we are not, so there is no reason to keep this guy around.  Kill him.
    pair<PDBSetPtr, long> whichPage = make_pair(me->getSet(), me->whichPage());
    allPages.erase(whichPage);

  } else {
    unpin(me, lock);
  }

  // log the down to zero
  logDownToZeroReferences(me->whichSet, me->whichPage());
}

bool PDBBufferManagerImpl::isRemovalStillValid(PDBPagePtr me) {

  // lock the page counter so we can work with it
  std::unique_lock<std::mutex> page_lck(me->lk);

  // ok if by some chance we made another reference while we were waiting for the lock
  // do not free it!
  if (me->refCount != 0) {
    return false;
  }

  // did we in the mean time started doing something with the page
  if (me->status == PDB_PAGE_LOADING || me->status == PDB_PAGE_UNLOADING) {
    return false;
  }

  // if the page is anonymous we are done here
  if (me->isAnonymous()) {
    return true;
  }

  pair<PDBSetPtr, long> whichPage = make_pair(me->getSet(), me->whichPage());
  auto it = allPages.find(whichPage);

  // was the page removed while we were waiting
  if (it == allPages.end()) {
    return false;
  }

  // was the page removed and then replaced by another one while we were waiting
  // (this is the last check it this works the removal is safe)
  return it->second.get() == me.get();
}

void PDBBufferManagerImpl::flushPage(PDBPagePtr &page, int32_t fd, unique_lock<mutex> &lock) {

  // the page is not unloading unlock the buffer manager so we don't stall
  lock.unlock();

  PDBPageInfo myInfo = page->getLocation();

  ssize_t write_bytes = pwrite(fd, page->getBytes(), MIN_PAGE_SIZE << myInfo.numBytes, myInfo.startPos);
  if (write_bytes == -1) {
    std::cerr << "error in createAdditionalMiniPages when writing page to disk with errno: " << strerror(errno)
              << std::endl;
    exit(1);
  }

  // lock the thing again
  lock.lock();
}

void PDBBufferManagerImpl::movePageFreeLocation(PDBPagePtr &page) {

  // get an empty page of the required size
  int32_t pageNum = 0;
  auto &emptyPages = emptyMiniPages[page->location.numBytes];
  for(int32_t i = 1; i < emptyPages.size(); i++) {
    if(emptyPages[i] < emptyPages[pageNum]){
      pageNum = i;
    }
  }

  // swap it to the back and fetch it
  std::swap(emptyPages[pageNum], emptyPages[emptyMiniPages[page->location.numBytes].size() - 1]);
  auto ep = emptyPages.back();
  emptyPages.pop_back();

  // get the parent page of the empty page
  auto parent = (char *) sharedMemory.memory + ((((char *) ep - (char *) sharedMemory.memory) / sharedMemory.pageSize) * sharedMemory.pageSize);

  // remove the empty page from the unused pages since we are using it now
  auto it = std::find(unusedMiniPages[parent].first.begin(),  unusedMiniPages[parent].first.end(), ep);
  unusedMiniPages[parent].first.erase(it);

  // do the copy and
  std::memcpy(ep, page->getBytes(), page->getSize());
  page->bytes = ep;
  page->status = PDB_PAGE_LOADED;

  // move the page into constituent pages
  constituentPages[parent].push_back(page);
}

// this is only called with a locked buffer manager
void PDBBufferManagerImpl::createAdditionalMiniPages(int64_t whichSize, unique_lock<mutex> &lock) {

  // if somebody else is making a mini page of the requested size wait here
  spaceCV.wait(lock, [&] { return !isCreatingSpace[whichSize]; });

  // did somebody create them while we were waiting
  if (!emptyMiniPages[whichSize].empty()) {
    return;
  }

  // mark that we are creating the page of the requested size
  isCreatingSpace[whichSize] = true;

  // first, we see if there is a page that we can break up; if not, then make one
  if (emptyFullPages.empty()) {

    // if there are no pages, give a fatal error
    if (lastUsed.empty()) {
      std::cerr << "This is really bad.  We seem to have run out of RAM in the storage manager.\n";
      std::cerr << "I suspect that there are too many pages pinned.\n";
      exit(1);
    }

    // find the LRU
    auto pageIt = lastUsed.begin();

    // we copy the thing so we can erase it from the lastUsed
    auto page = *pageIt;

    // remove the evicted page from the queue this prevents other threads from using it
    lastUsed.erase(pageIt);

    // mark all pages as unloading
    std::for_each(constituentPages[page.first].begin(),
                  constituentPages[page.first].end(),
                  [](auto &a) { a->status = PDB_PAGE_UNLOADING; });

    // remove the unused pages
    auto &unused = unusedMiniPages[page.first];
    auto &emptyPages = emptyMiniPages[unused.second];

    // go through each unused mini page and remove it!
    for (auto const &kt : unused.first) {
      auto const jt = std::find(emptyPages.begin(), emptyPages.end(), kt);
      emptyPages.erase(jt);
    }

    // clear the unused pages
    unused.first.clear();

    // now let all of the constituent pages know the RAM is no longer usable
    // this loop is safe since nobody can access it since we removed the page from lastUsed
    for (auto &a: constituentPages[page.first]) {

      if (a->isAnonymous() && a->isDirty()) {

        if (availablePositions[a->getLocation().numBytes].empty()) {

          a->getLocation().startPos = lastTempPos;
          lastTempPos += (MIN_PAGE_SIZE << a->getLocation().numBytes);

        } else {

          a->getLocation().startPos = availablePositions[a->getLocation().numBytes].back();
          availablePositions[a->getLocation().numBytes].pop_back();
        }

        // do we have some pages we can move this page to
        if(!emptyMiniPages[a->location.numBytes].empty()) {

          // move the page to a free location so that we can delay flushing it
          movePageFreeLocation(a);

          // we moved it go to the next page
          continue;
        }
        else {

          // flush the page
          flushPage(a, tempFileFD, lock);
        }

      } else {

        // check if the page is dirty if it is we either need to flush it or move it to a new location
        if (a->isDirty()) {

          // do we have some pages we can move this page to
          if(!emptyMiniPages[a->location.numBytes].empty()) {

            // move the page to a free location so that we can delay flushing it
            movePageFreeLocation(a);

            // we moved it go to the next page
            continue;
          }
          else {

            // flush the page to disk
            flushPage(a, fds[a->getSet()], lock);
          }
        }

        // lock the page so we can check the references
        a->lk.lock();

        // if the number of outstanding references is zero, just kill it
        if (a->numRefs() == 0) {

          pair<PDBSetPtr, long> whichPage = make_pair(a->getSet(), a->whichPage());
          allPages.erase(whichPage);
        }

        // unlock the page since we are done with checking the references
        a->lk.unlock();
      }

      a->setClean();
      a->setBytes(nullptr);
    }

    // mark all pages as not loaded
    std::for_each(constituentPages[page.first].begin(),
                  constituentPages[page.first].end(),
                  [](auto &a) { if(a->getBytes() == nullptr) { a->status = PDB_PAGE_NOT_LOADED; } });

    // notify all the threads that are paused because of a status
    pagesCV.notify_all();

    // and erase the page
    constituentPages[page.first].clear();
    emptyFullPages.push_back(page.first);
    numPinned.erase(page.first);
  }

  // now, we have a big page, so we can break it up into mini-pages
  size_t inc = MIN_PAGE_SIZE << whichSize;
  auto &unused = unusedMiniPages[emptyFullPages.back()];
  unused.second = whichSize;

  for (size_t offset = 0; offset < sharedMemory.pageSize; offset += inc) {

    // store the empty mini page as a mini page of that size
    emptyMiniPages[whichSize].push_back(((char *) emptyFullPages.back()) + offset);

    // store the mini page as unused
    unused.first.emplace_back(((char *) emptyFullPages.back()) + offset);
  }

  // set the number of pinned pages to zero... we will always pin this page subsequently,
  // so no need to insert into the LRU queue
  numPinned[emptyFullPages.back()] = 0;

  // and get rid of the empty full page
  emptyFullPages.pop_back();

  isCreatingSpace[whichSize] = false;
  spaceCV.notify_all();
}

void PDBBufferManagerImpl::freezeSize(PDBPagePtr me, size_t numBytes) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lock(m);

  // do the actual freezing
  freezeSize(me, numBytes, lock);

  // log the freeze size
  logFreezeSize(me->whichSet, me->whichPage(), numBytes);
}

void PDBBufferManagerImpl::freezeSize(PDBPagePtr me, size_t numBytes, unique_lock<mutex> &lock) {

  // you cannot freeze an unpinned page
  if(!me->isPinned()) {
    std::cerr << "You cannot freeze an unpinned page.\n";
    exit(1);
  }

  // you cannot freeze the size of a page twice
  if (me->sizeIsFrozen()) {
    std::cerr << "You cannot freeze the size of a page twice.\n";
    exit(1);
  }

  // set the indicator of the page to frozen
  me->freezeSize();

  // figure out the size
  size_t bytesRequired = getLogPageSize(numBytes);
  if(bytesRequired > me->getLocation().numBytes) {
    std::cerr << "You cannot freeze to a size of a page grater than the size you requested!\n";
    exit(-1);
  }

  // set the page size
  me->getLocation().numBytes = bytesRequired;

  // since we are freezing we need to break up the large page into smaller pages, so we can use the space
  size_t inc = MIN_PAGE_SIZE << bytesRequired;
  auto &unused = unusedMiniPages[me->bytes];
  unused.second = bytesRequired;

  for (size_t offset = inc; offset < sharedMemory.pageSize; offset += inc) {

    // store the empty mini page as a mini page of that size
    emptyMiniPages[bytesRequired].push_back(((char *) me->bytes) + offset);

    // store the mini page as unused
    unused.first.emplace_back(((char *) me->bytes) + offset);
  }
}

void PDBBufferManagerImpl::unpin(PDBPagePtr me) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lock(m);

  // unlock the page
  unpin(me, lock);

  // log the unpin
  logUnpin(me->whichSet, me->whichPage());
}

void PDBBufferManagerImpl::unpin(PDBPagePtr me, unique_lock<mutex> &lock) {

  // if it is not pinned no need to unpin it..
  if (!me->isPinned()) {
    return;
  }

  // it is no longer pinned
  me->setUnpinned();

  // freeze the size, if needed
  if (!me->sizeIsFrozen()) {
    me->freezeSize();
  }

  // first, we find the parent of this guy
  void *memLoc = (char *) sharedMemory.memory
      + ((((char *) me->getBytes() - (char *) sharedMemory.memory) / sharedMemory.pageSize) * sharedMemory.pageSize);

  // and decrement the number of pinned minipages
  if (numPinned[memLoc] > 0) {
    numPinned[memLoc]--;
  }

  // if the number of pinned minipages is now zero, put into the LRU structure
  if (numPinned[memLoc] == 0) {

    // add it to the LRU
    numPinned[memLoc] = -lastTimeTick;

    // add the physical page to the LRU
    lastUsed.insert(make_pair(memLoc, lastTimeTick));

    // increment the time tick
    lastTimeTick++;
  }

  // now that the page is unpinned, we find a physical location for it
  pair<PDBSetPtr, long> whichPage = make_pair(me->getSet(), me->whichPage());
  if (!me->isAnonymous()) {

    // if we don't know where to write it, figure it out
    if (pageLocations.find(whichPage) == pageLocations.end()) {
      me->getLocation().startPos = endOfFiles[me->getSet()];
      pair<PDBSetPtr, size_t> myIndex = make_pair(me->getSet(), me->whichPage());
      pageLocations[whichPage] = me->getLocation();
      endOfFiles[me->getSet()] += (MIN_PAGE_SIZE << me->getLocation().numBytes);
    }
  }
}

void PDBBufferManagerImpl::pinParent(const PDBPagePtr &me) {

  // first, we determine the parent of this guy
  void *whichPage = (char *) sharedMemory.memory
      + ((((char *) me->getBytes() - (char *) sharedMemory.memory) / sharedMemory.pageSize) * sharedMemory.pageSize);

  // and increment the number of pinned minipages
  if (numPinned[whichPage] < 0) {
    pair<void *, unsigned> id = make_pair(whichPage, -numPinned[whichPage]);
    lastUsed.erase(id);
    numPinned[whichPage] = 1;
  } else {
    numPinned[whichPage]++;
  }
}

void PDBBufferManagerImpl::repin(PDBPagePtr me) {

  // lock the buffer manager
  unique_lock<mutex> lock(m);

  // wait while the page is loading
  pagesCV.wait(lock, [&] { return !(me->status == PDB_PAGE_LOADING || me->status == PDB_PAGE_UNLOADING); });

  // call the actual repin function
  repin(me, lock);
}

void PDBBufferManagerImpl::repin(PDBPagePtr me, unique_lock<mutex> &lock) {

  // first, we need to see if this page is currently pinned
  if (me->isPinned()) {
    return;
  }

  // it is not currently pinned, so see if it is in RAM
  if (me->getBytes() != nullptr) {

    // if it is a whole a page just repin and return, otherwise we need to check for fragmentation
    if(me->getSize() == getMaxPageSize()) {

      // it is currently pinned, so mark the parent as pinned
      me->setPinned();
      pinParent(me);
      return;
    }

    // determine the parent of this guy
    void *currentParent = (char *) sharedMemory.memory
        + ((((char *) me->getBytes() - (char *) sharedMemory.memory) / sharedMemory.pageSize) * sharedMemory.pageSize);

    // figure out how many pages are there that are on the same parent page as this page
    // but are not used
    auto numUnusedPages = unusedMiniPages[currentParent].first.size();

    // check if this is the only page on the parent page
    if((sharedMemory.pageSize / me->getSize()) != (numUnusedPages + 1)){

      // if it is not then just repin
      me->setPinned();
      pinParent(me);
      return;
    }

    // if it is the only page check if we have some pages of the same size that are not
    // on this parent page
    void* ep = nullptr;
    std::for_each(emptyMiniPages[me->location.numBytes].begin(), emptyMiniPages[me->location.numBytes].end(), [&](void * &page) {

      // figure out the parent page of the empty mini page
      void *pp = (char *) sharedMemory.memory + ((((char *) page - (char *) sharedMemory.memory) / sharedMemory.pageSize) * sharedMemory.pageSize);

      // skip this
      if(pp == currentParent)  { return; }

      // we always take the page with the minimum page
      if(ep != nullptr) { ep = std::min(ep, page); } else { ep = page; }
    });

    if(ep != nullptr) {

      // figure out the parent page of the empty mini page
      void *pp = (char *) sharedMemory.memory + ((((char *) ep - (char *) sharedMemory.memory) / sharedMemory.pageSize) * sharedMemory.pageSize);

      // ok they are not on the same page move it there
      std::memcpy(ep, me->getBytes(), me->getSize());

      // now, add him to the list of constituent pages
      constituentPages[pp].push_back(me);

      // remove the page from the constituentPages of the previous parent page
      constituentPages[currentParent].clear();

      // remove the empty pages associated with this page
      for(auto up : unusedMiniPages[currentParent].first) {

        // remove the unused page
        auto miniPage = std::find(emptyMiniPages[me->location.numBytes].begin(), emptyMiniPages[me->location.numBytes].end(), up);
        emptyMiniPages[me->location.numBytes].erase(miniPage);
      }

      // clear the unused pages from the parent because they don't exist it is a whole page now
      unusedMiniPages[currentParent].first.clear();

      // mark the parent page as free
      emptyFullPages.push_back(currentParent);

      // remove it from the LRU
      if(numPinned[currentParent] < 0) {

        // remvo the physical page from the LRU
        lastUsed.erase(lastUsed.find(make_pair(currentParent, -numPinned[currentParent])));
      }

      // set the number of pinned to zero
      numPinned[currentParent] = 0;

      // remove the mini page we just used
      auto miniPage = std::find(emptyMiniPages[me->location.numBytes].begin(), emptyMiniPages[me->location.numBytes].end(), ep);
      emptyMiniPages[me->location.numBytes].erase(miniPage);

      // remove the unused mini page
      auto &unused = unusedMiniPages[pp].first;
      auto it = std::find(unused.begin(), unused.end(), ep);
      unused.erase(it);

      // update the bytes
      me->bytes = ep;

      // this guy is now pinned
      me->setPinned();
      pinParent(me);
      return;
    }

    // if we could not find a place to move it just repin
    me->setPinned();
    pinParent(me);
    return;
  }

  // it is an anonymous page, so we have to look up its location
  PDBPageInfo myInfo = me->getLocation();

  // set the status to loading
  me->status = PDB_PAGE_LOADING;

  // grab space from an empty page
  me->setBytes(getEmptyMemory(myInfo.numBytes, lock));

  registerMiniPage(me);
  me->setPinned();

  if (me->isAnonymous()) {

    // unlock the buffer manager
    lock.unlock();

    // read the page from disk
    ssize_t read_bytes;
    read_bytes = pread(tempFileFD, me->getBytes(), MIN_PAGE_SIZE << myInfo.numBytes, myInfo.startPos);
    if (read_bytes == -1) {
      std::cerr << "repin error when reading anonymous page from disk with errno: " << strerror(errno) << std::endl;
      exit(1);
    }

  } else {

    // unlock the buffer manager
    lock.unlock();

    // read the page from disk
    ssize_t read_bytes;
    read_bytes = pread(fds[me->getSet()], me->getBytes(), MIN_PAGE_SIZE << myInfo.numBytes, myInfo.startPos);
    if (read_bytes == -1) {
      std::cerr << "repin error when reading the page from disk with errno: " << strerror(errno) << std::endl;
      exit(1);
    }
  }

  // unlock the mutex
  lock.lock();

  // set the page to loaded
  me->status = PDB_PAGE_LOADED;

  // log the repin
  logRepin(me->whichSet, me->whichPage());

  // notify all waiting conditional variables
  lock.unlock();
  pagesCV.notify_all();
}

PDBPageHandle PDBBufferManagerImpl::getPage() {
  return getPage(sharedMemory.pageSize);
}

PDBPageHandle PDBBufferManagerImpl::getPage(size_t maxBytes) {

  if (!initialized) {
    cerr << "Can't call getMaxPageSize () without initializing the storage manager\n";
    exit(1);
  }

  if (maxBytes > sharedMemory.pageSize) {
    std::cerr << maxBytes << " is larger than the system page size of " << sharedMemory.pageSize << "\n";
  }

  // lock the buffer manager
  unique_lock<mutex> lock(m);

  // figure out the size of the page that we need
  size_t bytesRequired = getLogPageSize(maxBytes);

  // grab space from an empty page
  void *space = getEmptyMemory(bytesRequired, lock);

  // figure out a free page number
  if (freeAnonPageNumbers.empty()) {

    // figure out a new page
    lastFreeAnonPageNumber++;

    // did we hit an overflow
    if (lastFreeAnonPageNumber == std::numeric_limits<uint64_t>::max()) {

      // log what happened
      getLogger()->fatal(
          "Too many anonymous pages " + std::to_string(lastFreeAnonPageNumber) + " were requested at the same time .");

      // kill the server.. we might consider a more graceful shutdown
      exit(-1);
    }

    // store the new free page
    freeAnonPageNumbers.emplace_back(lastFreeAnonPageNumber);
  }

  // grab a page number
  uint64_t pageNum = freeAnonPageNumbers.back();
  freeAnonPageNumbers.pop_back();

  PDBPagePtr returnVal = make_shared<PDBPage>(*this);
  returnVal->setMe(returnVal);
  returnVal->setPinned();
  returnVal->setDirty();
  returnVal->setBytes(space);
  returnVal->setAnonymous(true);
  returnVal->setPageNum(pageNum);
  returnVal->getLocation().numBytes = bytesRequired;
  returnVal->getLocation().startPos = -1;
  returnVal->status = PDB_PAGE_LOADED;
  registerMiniPage(returnVal);

  // log the get page
  logGetPage(maxBytes, returnVal->whichPage());

  // return
  return make_shared<PDBPageHandleBase>(returnVal);
}

PDBPageHandle PDBBufferManagerImpl::getPage(PDBSetPtr whichSet, uint64_t i) {

  if (!initialized) {
    cerr << "Can't call getMaxPageSize () without initializing the storage manager\n";
    exit(1);
  }

  // make sure we don't have a null table
  if (whichSet == nullptr) {
    cerr << "Can't allocate a page with a null table!!\n";
    exit(1);
  }

  // check if we have the file for this set...
  checkIfOpen(whichSet);

  // lock the buffer manager
  std::unique_lock<std::mutex> lock(m);

  // next, see if the page is already in existence
  pair<PDBSetPtr, size_t> whichPage = make_pair(whichSet, i);
  if (allPages.find(whichPage) == allPages.end()) {

    // it is not there, so see if we have previously created it
    if (pageLocations.find(whichPage) == pageLocations.end()) {

      // we have not previously created it
      PDBPageInfo myInfo;
      myInfo.startPos = i;
      myInfo.numBytes = logOfPageSize;

      // create the page and store it in the allPages, we have to do this before we unlock the buffer manager,
      // and lock the page. This is to avoid the the scenario where some other thread requests this page but we still haven't
      // finished creating it...
      auto page = make_shared<PDBPage>(*this);
      page->setMe(page);
      page->setPinned();
      page->setDirty();
      page->setSet(whichSet);
      page->setPageNum(i);
      page->setAnonymous(false);
      page->getLocation() = myInfo;

      // store it in allPages
      allPages[whichPage] = page;

      // mark that we are loading the page
      page->status = PDB_PAGE_LOADING;

      // set the physical address of the page
      page->setBytes(getEmptyMemory(myInfo.numBytes, lock));

      // and now that we have the physical we can simply register the page (add it to the constituent pages and pin the parent)
      registerMiniPage(page);

      // make a return value
      auto pageHandle = make_shared<PDBPageHandleBase>(page);

      // mark the page as loaded
      page->status = PDB_PAGE_LOADED;

      // log the get page
      logGetPage(whichSet, i);

      // notify all waiting conditional variables
      lock.unlock();
      pagesCV.notify_all();

      // return page handle
      return pageHandle;

    } else {

      // we have previously created it, so load it up
      PDBPageInfo &myInfo = pageLocations[whichPage];

      // create the page and store it in the allPages, we have to do this before we unlock the buffer manager,
      // and lock the page. This is to avoid the the scenario where some other thread requests this page but we still haven't
      // finished creating it
      auto page = make_shared<PDBPage>(*this);
      page->setMe(page);
      page->setPinned();
      page->setClean();
      page->setSet(whichSet);
      page->setPageNum(i);
      page->setAnonymous(false);
      page->getLocation() = myInfo;

      // store it in allPages
      allPages[whichPage] = page;

      // mark that we are loading the page
      page->status = PDB_PAGE_LOADING;

      // make a return value
      auto pagerHandle = make_shared<PDBPageHandleBase>(page);

      // grab space from an empty page
      void *space = getEmptyMemory(myInfo.numBytes, lock);

      // set the physical address of the page
      page->setBytes(space);

      // and now that we have the physical we can simply register the page (add it to the constituent pages and pin the parent)
      registerMiniPage(page);

      // unlock since we are working on loading the file
      lock.unlock();

      // read the data from disk
      auto fd = getFileDescriptor(whichSet);

      ssize_t read_bytes;
      read_bytes = pread(fd, space, MIN_PAGE_SIZE << myInfo.numBytes, myInfo.startPos);
      if (read_bytes == -1) {
        std::cerr << "getPage Error when read data from disk with errno: " << strerror(errno) << std::endl;
        exit(1);
      }

      // finished working on the thing lock it
      lock.lock();

      // mark the page as loaded
      page->status = PDB_PAGE_LOADED;

      // log the get page
      logGetPage(whichSet, i);

      // notify all waiting conditional variables
      lock.unlock();
      pagesCV.notify_all();

      // return the page handle
      return pagerHandle;
    }
  }

  // grab a page
  auto page = allPages[whichPage];

  // make a handle to the page we have to do that before we lock the conditional variable so the page does not get
  // removed from the allPages if it was unloading and there are no handles to it...
  auto ret = make_shared<PDBPageHandleBase>(page);

  // wait while the page is loading
  pagesCV.wait(lock, [&] { return !(page->status == PDB_PAGE_LOADING || page->status == PDB_PAGE_UNLOADING); });

  // log the get page
  logGetPage(whichSet, i);

  // it is there, so return it
  repin(page, lock);

  return ret;
}

void *PDBBufferManagerImpl::getEmptyMemory(int64_t pageSize, unique_lock<mutex> &lock) {

  // get space for it... first see if the space is available
  if (emptyMiniPages[pageSize].empty()) {
    createAdditionalMiniPages(pageSize, lock);
  }

  // grab an empty page
  void *space = emptyMiniPages[pageSize].back();
  emptyMiniPages[pageSize].pop_back();

  // determine the parent of this guy
  void *whichPage = (char *) sharedMemory.memory
      + ((((char *) space - (char *) sharedMemory.memory) / sharedMemory.pageSize) * sharedMemory.pageSize);

  // remove the page we just grabbed from the pool of the unused pages for the parent page...
  auto &parentPage = unusedMiniPages[whichPage].first;
  auto it = std::find(parentPage.begin(), parentPage.end(), space);
  parentPage.erase(it);

  return space;
}

int PDBBufferManagerImpl::getFileDescriptor(const PDBSetPtr &whichSet) {

  // lock the file descriptors structure to grab a descriptor
  unique_lock<mutex> blockLck(fdLck);

  auto fd = fds[whichSet];
  return fd;
}

void PDBBufferManagerImpl::checkIfOpen(const PDBSetPtr &whichSet) {

  unique_lock<mutex> blockLck(fdLck);

  // open the file, if it is not open
  if (fds.find(whichSet) == fds.end()) {

    // open the file
    string fileLoc = storageLoc + "/" + whichSet->getSetName() + "." + whichSet->getDBName();
    int fd = open(fileLoc.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd != -1) {
      fds[whichSet] = fd;
    } else {
      std::cerr << "Fail to open the file at " << fileLoc << std::endl;
      exit(1);
    }
    // init the end of the file if we just created a new file
    if (endOfFiles.find(whichSet) == endOfFiles.end()) {
      endOfFiles[whichSet] = 0;
    }
  }
}
size_t PDBBufferManagerImpl::getLogPageSize(size_t numBytes) {

  size_t bytesRequired = 0;
  size_t curSize = MIN_PAGE_SIZE;
  for (; curSize < numBytes; curSize = (curSize << 1)) {
    bytesRequired++;
  }

  return bytesRequired;
}


}

#endif