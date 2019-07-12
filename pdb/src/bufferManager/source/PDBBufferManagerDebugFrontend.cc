#ifdef DEBUG_BUFFER_MANAGER

#include "PDBBufferManagerDebugFrontend.h"

namespace pdb {

const uint64_t PDBBufferManagerDebugFrontend::DEBUG_MAGIC_NUMBER = 10202026;

void PDBBufferManagerDebugFrontend::initDebug(const std::string &timelineDebugFile) {

  // open debug file
  debugTimelineFile = open(timelineDebugFile.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);

  // check if we actually opened the file
  if (debugTimelineFile == 0) {
    exit(-1);
  }

  // write out the magic number
  write(debugTimelineFile, &DEBUG_MAGIC_NUMBER, sizeof(DEBUG_MAGIC_NUMBER));

  // write out the number of pages
  write(debugTimelineFile, &sharedMemory.numPages, sizeof(sharedMemory.numPages));

  // write out the page size
  write(debugTimelineFile, &sharedMemory.pageSize, sizeof(sharedMemory.pageSize));
}

void PDBBufferManagerDebugFrontend::logTimeline() {

  // just a temp value
  uint64_t tmp;

  // lock the timeline file
  std::unique_lock<std::mutex> lck(m);

  // increment the debug tick
  uint64_t tick = debugTick++;

  // write out the tick
  write(debugTimelineFile, &tick, sizeof(tick));

  // write out the number of pages
  uint64_t numPages = 0;
  for(const auto &pages : constituentPages) {

    // go through the mini pages on the page
    for(const auto &page : pages.second) {

      // if it is not unloading add it
      if(page->getBytes() != nullptr) {
        numPages++;
      }
    }
  }
  write(debugTimelineFile, &numPages, sizeof(numPages));

  // write out the page info
  for(const auto &pages : constituentPages) {

    // write out all the mini pages
    for(const auto &page : pages.second) {

      // if the page is not unloading we skip it
      if(page->getBytes() == nullptr) {
        continue;
      }

      // if this is an not anonmous page
      if(page->getSet() != nullptr) {

        // write out the database name
        tmp = page->getSet()->getDBName().size();
        write(debugTimelineFile, &tmp, sizeof(tmp));
        write(debugTimelineFile, page->getSet()->getDBName().c_str(), tmp);

        // write out the set name
        tmp = page->getSet()->getSetName().size();
        write(debugTimelineFile, &tmp, sizeof(tmp));
        write(debugTimelineFile, page->getSet()->getSetName().c_str(), tmp);
      } else {

        // write out zeros twice, meaning both strings are empty
        tmp = 0;
        write(debugTimelineFile, &tmp, sizeof(tmp));
        write(debugTimelineFile, &tmp, sizeof(tmp));
      }

      // write out the page number
      tmp = page->whichPage();
      write(debugTimelineFile, &tmp, sizeof(tmp));

      uint64_t offset = (uint64_t) page->getBytes() - (uint64_t)sharedMemory.memory;
      write(debugTimelineFile, &offset, sizeof(offset));

      // grab the page size
      tmp = page->getSize();
      write(debugTimelineFile, &tmp, sizeof(tmp));
    }
  }

  // write out the number of empty pages
  uint64_t numEmptyPages = emptyFullPages.size();
  for(const auto &miniPages : emptyMiniPages) { numEmptyPages += miniPages.size();}
  write(debugTimelineFile, &numEmptyPages, sizeof(numEmptyPages));

  // write out the empty full pages
  for(const auto &emptyFullPage : emptyFullPages) {

    // figure out the offset
    uint64_t offset = (uint64_t)emptyFullPage - (uint64_t)sharedMemory.memory;
    write(debugTimelineFile, &offset, sizeof(offset));

    // empty full page has the maximum page size
    write(debugTimelineFile, &sharedMemory.pageSize, sizeof(sharedMemory.pageSize));
  }

  // write out the empty mini pages
  for(auto i = 0; i < emptyMiniPages.size(); ++i) {

    // figure out the size of the page
    uint64_t pageSize = MIN_PAGE_SIZE << i;

    // write out the mini pages of this size
    for(const auto &emptyPage : emptyMiniPages[i]) {

      // figure out the offset
      uint64_t offset = (uint64_t)emptyPage - (uint64_t)sharedMemory.memory;
      write(debugTimelineFile, &offset, sizeof(offset));

      // empty full page has the maximum page size
      write(debugTimelineFile, &pageSize, sizeof(pageSize));
    }
  }
}

void PDBBufferManagerDebugFrontend::logGetPage(const PDBSetPtr &whichSet, uint64_t i) {

  // log the timeline
  logTimeline();
}

void PDBBufferManagerDebugFrontend::logGetPage(size_t minBytes) {

  // log the timeline
  logTimeline();
}

void PDBBufferManagerDebugFrontend::logFreezeSize(const PDBPagePtr &me, size_t numBytes) {

  // log the timeline
  logTimeline();
}

void PDBBufferManagerDebugFrontend::logUnpin(const PDBPagePtr &me) {

  // log the timeline
  logTimeline();
}

void PDBBufferManagerDebugFrontend::logRepin(const PDBPagePtr &me) {

  // log the timeline
  logTimeline();
}

void PDBBufferManagerDebugFrontend::logFreeAnonymousPage(const PDBPagePtr &me) {

  // log the timeline
  logTimeline();
}

void PDBBufferManagerDebugFrontend::logDownToZeroReferences(const PDBPagePtr &me) {

  // log the timeline
  logTimeline();
}

void PDBBufferManagerDebugFrontend::logClearSet(const PDBSetPtr &set) {

  // log the timeline
  logTimeline();
}

}

#endif