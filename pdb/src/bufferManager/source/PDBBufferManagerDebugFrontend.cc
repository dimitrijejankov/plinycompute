#include "PDBBufferManagerDebugFrontend.h"

namespace pdb {

const uint64_t PDBBufferManagerDebugFrontend::DEBUG_MAGIC_NUMBER = 10202026;

void PDBBufferManagerDebugFrontend::initDebug(const std::string &timelineDebugFile) {

  // open debug file
  debugTimelineFile = open(timelineDebugFile.c_str(), O_CREAT | O_RDWR, 0666);

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

  // write out the number of pages and the number of empty slots
  uint64_t numPages = allPages.size();
  write(debugTimelineFile, &numPages, sizeof(numPages));

  // write out the page info
  for(const auto &page : allPages) {

    // write out the database name
    tmp = page.first.first->getDBName().size();
    write(debugTimelineFile, &tmp, sizeof(tmp));
    write(debugTimelineFile, page.first.first->getDBName().c_str(), tmp);

    // write out the set name
    tmp = page.first.first->getSetName().size();
    write(debugTimelineFile, &tmp, sizeof(tmp));
    write(debugTimelineFile, page.first.first->getSetName().c_str(), tmp);

    // write out the page number
    tmp = page.first.second;
    write(debugTimelineFile, &tmp, sizeof(tmp));

    // grab the offset
    int64_t offset = page.second->getBytes() != nullptr ? (int64_t)page.second->getBytes() - (int64_t)sharedMemory.memory : -1;
    write(debugTimelineFile, &offset, sizeof(offset));

    // grab the page size
    tmp = page.second->getSize();
    write(debugTimelineFile, &tmp, sizeof(tmp));
  }

  // write out the number of empty mini pages
  uint64_t numEmptyPages = 0;
  for(const auto &miniPages : emptyMiniPages) { numEmptyPages += miniPages.size();}
  write(debugTimelineFile, &numEmptyPages, sizeof(numEmptyPages));

  // write out the empty full pages
  for(auto i = 0; i < emptyMiniPages.size(); ++i) {

    // figure out the size of the page
    uint64_t pageSize = MIN_PAGE_SIZE << i;

    // write out the mini pages of this size
    for(const auto &emptyPage : emptyMiniPages[i]) {

      // figure out the offset
      int64_t offset = (int64_t)emptyPage - (int64_t)sharedMemory.memory;
      write(debugTimelineFile, &offset, sizeof(offset));

      // empty full page has the maximum page size
      write(debugTimelineFile, &sharedMemory.pageSize, sizeof(sharedMemory.pageSize));
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
