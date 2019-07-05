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

protected:

  void logGetPage(const PDBSetPtr &whichSet, uint64_t i) override;
  void logGetPage(size_t minBytes) override;
  void logFreezeSize(const PDBPagePtr &me, size_t numBytes) override;
  void logUnpin(const PDBPagePtr &me) override;
  void logRepin(const PDBPagePtr &me) override;
  void logFreeAnonymousPage(const PDBPagePtr &me) override;
  void logDownToZeroReferences(const PDBPagePtr &me) override;
  void logClearSet(const PDBSetPtr &set) override;

  void initDebug(const std::string &timelineDebugFile);

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
   * 8 bytes as "numUnused" - the number of unused mini pages
   *
   * every page that is not listed here or in the previous allocated pages  is not used.
   * numUnused times ( 8 bytes signed for the offset of the unused page |  8 bytes for the size of the page)
   *
   */
  void logTimeline();

  /**
   * The lock we made
   */
  std::mutex m;

  /**
   * The tick so we can order events
   */
  uint64_t debugTick = 0;

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
