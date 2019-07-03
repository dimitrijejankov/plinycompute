#pragma once

#include "PDBBufferManagerFrontEnd.h"

#include <boost/filesystem/path.hpp>
#include <fcntl.h>

namespace pdb {

namespace fs = boost::filesystem;

class PDBBufferManagerDebugFrontend : public PDBBufferManagerFrontEnd {
public:

  PDBBufferManagerDebugFrontend(const string &tempFileIn,
                                size_t pageSizeIn,
                                size_t numPagesIn,
                                const string &metaFile,
                                const string &storageLocIn) : PDBBufferManagerFrontEnd(tempFileIn,
                                                                                       pageSizeIn,
                                                                                       numPagesIn,
                                                                                       metaFile,
                                                                                       storageLocIn) {
    // the location of the debug file
    string fileLoc = storageLoc + "/debug.dt";

    // init the debug file
    initDebug(fileLoc);
  }

  explicit PDBBufferManagerDebugFrontend(const NodeConfigPtr &config) : PDBBufferManagerFrontEnd(config) {

    // create the root directory
    fs::path dataPath(config->rootDirectory);
    dataPath.append("/data");

    // init the debug file
    initDebug((dataPath / "debug.dt").string());
  }

private:

  void initDebug(const std::string &timelineDebugFile) {


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

  /**
   * Writes out the state of the buffer manager at this time
   * The layout of the state is like this (all values are unsigned little endian unless specified otherwise):
   *
   * 8 bytes as "tick" - the current tick, which can tell us the order of the things that happened
   *
   * 8 bytes as "numPages"- the number of pages in the buffer manager, both the ones in ram and the ones outside
   *
   * numPages times the following - | dbName | setName | pageNum | offset | page size |
   *
   * The values here are the following :
   *
   * dbName is a string and has the following layout | 32 bit for string size | cstring of specified size |
   * setName is a string and has the following layout | 32 bit for string size | cstring of specified size |
   * pageNum is 8 bytes and indicates the number of the page within the set
   * offset is 8 bytes signed, it is -1 if the page is not loaded, otherwise it is the offset from the start of the memory
   * page size is 8 bytes, indicates the size of the page
   *
   * 8 bytes as "numUnused" - the number of unused mini pages.
   * numUnused times ( 8 bytes signed for the offset of the unused page |  8 bytes for the size of the page)
   *
   */
  void logTimeline() {

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

      uint64_t tmp;

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
    }

  }

  /**
   * The lock we made
   */
  std::mutex m;

  /**
   * The tick so we can order events
   */
  atomic_uint64_t debugTick;

  /**
   * The file we are going to write all the debug timeline files
   */
  int debugTimelineFile = 0;

  /**
   * The magic number the debug files start with
   */
  static const uint64_t DEBUG_MAGIC_NUMBER;
};

}
