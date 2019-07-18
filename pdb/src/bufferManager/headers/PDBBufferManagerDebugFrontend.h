#pragma once

#ifdef DEBUG_BUFFER_MANAGER

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
    // init the debug file
    initDebug(storageLoc + "/debug.dt", storageLoc + "/debugSymbols.ds", storageLoc + "/stackTraces.dst");
  }

  explicit PDBBufferManagerDebugFrontend(const NodeConfigPtr &config) : PDBBufferManagerFrontEnd(config) {

    // create the root directory
    fs::path dataPath(config->rootDirectory);
    dataPath.append("/data");

    // init the debug stuff
    initDebug((dataPath / "debug.dt").string(), (dataPath / "debugSymbols.ds").string(), (dataPath / "stackTraces.dst").string());
  }

  void registerHandlers(PDBServer &forMe) override;

  PDBBufferManagerInterfacePtr getBackEnd() override;

protected:

  enum class BufferManagerOperationType {
    GET_PAGE,
    FREEZE,
    UNPIN,
    REPIN,
    FREE,
    CLEAR,
    FORWARD,
    HANDLE_GET_PAGE,
    HANDLE_RETURN_PAGE,
    HANDLE_FREEZE_SIZE,
    HANDLE_PIN_PAGE,
    HANDLE_UNPIN_PAGE
  };

  struct traceHasher {
    std::size_t operator()(std::vector<size_t> const& c) const  {

      std::size_t seed = 0;
      for(const auto &value : c) {
        seed ^= value + 0x9e3779b9 + (seed<< 6) + (seed>>2);
      }

      return seed;
    }
  };

  void logOperation(uint64_t timestamp,
                    BufferManagerOperationType operation,
                    const std::string &dbName,
                    const std::string &setName,
                    uint64_t pageNumber,
                    uint64_t value,
                    uint64_t backendID);

  void logGetPage(const PDBSetPtr &whichSet, uint64_t i) override;
  void logGetPage(size_t minBytes, uint64_t pageNumber) override;
  void logFreezeSize(const PDBSetPtr &setPtr, size_t pageNum, size_t numBytes) override;
  void logUnpin(const PDBSetPtr &setPtr, size_t pageNum) override;
  void logRepin(const PDBSetPtr &setPtr, size_t pageNum) override;
  void logFreeAnonymousPage(uint64_t pageNumber) override;
  void logDownToZeroReferences(const PDBSetPtr &setPtr, size_t pageNum) override;
  void logClearSet(const PDBSetPtr &set) override;
  void logForward(const Handle<pdb::BufForwardPageRequest> &request) override;

 protected:

  void initDebug(const std::string &timelineDebugFile,
                 const std::string &debugSymbols,
                 const std::string &stackTraces);

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
  void logTimeline(const uint64_t &tick);

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
   * The file we are going to write all the debug symbols for the stack trace
   */
  int debugSymbolTableFile = 0;

  /**
   * The file where we are going to write all the stack traces
   */
  int stackTracesTableFile = 0;

  /**
   * The set of stack traces
   */
  std::unordered_map<std::vector<size_t>, size_t, traceHasher> stackTraces;

  /**
   * The magic number the debug files start with
   */
  static const uint64_t DEBUG_MAGIC_NUMBER;
};

}

#endif