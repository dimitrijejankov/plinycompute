#ifdef DEBUG_BUFFER_MANAGER

#include "PDBBufferManagerDebugFrontend.h"
#include "PDBBufferManagerDebugBackEnd.h"
#include <boost/stacktrace.hpp>
#include <HeapRequestHandler.h>

namespace pdb {

const uint64_t PDBBufferManagerDebugFrontend::DEBUG_MAGIC_NUMBER = 10202026;

namespace bs = boost::stacktrace;


void PDBBufferManagerDebugFrontend::initDebug(const std::string &timelineDebugFile,
                                              const std::string &debugSymbols,
                                              const std::string &stackTraces) {

  // open debug file
  debugTimelineFile = open(timelineDebugFile.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);

  // check if we actually opened the file
  if (debugTimelineFile == 0) {
    exit(-1);
  }

  // open symbol file
  debugSymbolTableFile = open(debugSymbols.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);

  // check if we actually opened the file
  if (debugSymbolTableFile == 0) {
    exit(-1);
  }

  // open stackTraces file
  stackTracesTableFile = open(stackTraces.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);

  // check if we actually opened the file
  if (stackTracesTableFile == 0) {
    exit(-1);
  }

  // write out the magic number
  write(debugTimelineFile, &DEBUG_MAGIC_NUMBER, sizeof(DEBUG_MAGIC_NUMBER));

  // write out the number of pages
  write(debugTimelineFile, &sharedMemory.numPages, sizeof(sharedMemory.numPages));

  // write out the page size
  write(debugTimelineFile, &sharedMemory.pageSize, sizeof(sharedMemory.pageSize));
}

void PDBBufferManagerDebugFrontend::logTimeline(const uint64_t &tick) {

  // just a temp value
  uint64_t tmp;

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

  // lock the timeline file
  std::unique_lock<std::mutex> lck(m);

  // increment the debug tick
  uint64_t tick = debugTick++;

  // log the get page operation
  logOperation(tick, BufferManagerOperationType::GET_PAGE, whichSet->getDBName(), whichSet->getSetName(), i, 0, 0);

  // log the timeline
  logTimeline(tick);
}

void PDBBufferManagerDebugFrontend::logGetPage(size_t minBytes, uint64_t pageNumber) {

  // lock the timeline file
  std::unique_lock<std::mutex> lck(m);

  // increment the debug tick
  uint64_t tick = debugTick++;

  // log the get page operation
  logOperation(tick, BufferManagerOperationType::GET_PAGE, "", "", pageNumber, minBytes, 0);

  // log the timeline
  logTimeline(tick);
}

void PDBBufferManagerDebugFrontend::logFreezeSize(const PDBSetPtr &setPtr, size_t pageNum, size_t numBytes) {

  // lock the timeline file
  std::unique_lock<std::mutex> lck(m);

  // increment the debug tick
  uint64_t tick = debugTick++;

  // get the database and set name
  std::string db = setPtr == nullptr ? "" : setPtr->getDBName();
  std::string set = setPtr == nullptr ? "" : setPtr->getSetName();

  // log the freeze operation
  logOperation(tick, BufferManagerOperationType::FREEZE, db, set, pageNum, numBytes, 0);

  // log the timeline
  logTimeline(tick);
}

void PDBBufferManagerDebugFrontend::logUnpin(const PDBSetPtr &setPtr, size_t pageNum) {

  // lock the timeline file
  std::unique_lock<std::mutex> lck(m);

  // increment the debug tick
  uint64_t tick = debugTick++;

  // get the database and set name
  std::string db = setPtr == nullptr ? "" : setPtr->getDBName();
  std::string set = setPtr == nullptr ? "" : setPtr->getSetName();

  // log the operation
  logOperation(tick, BufferManagerOperationType::UNPIN, db, set, pageNum, 0, 0);

  // log the timeline
  logTimeline(tick);
}

void PDBBufferManagerDebugFrontend::logRepin(const PDBSetPtr &setPtr, size_t pageNum) {

  // lock the timeline file
  std::unique_lock<std::mutex> lck(m);

  // increment the debug tick
  uint64_t tick = debugTick++;

  // get the database and set name
  std::string db = setPtr == nullptr ? "" : setPtr->getDBName();
  std::string set = setPtr == nullptr ? "" : setPtr->getSetName();

  // log the operation
  logOperation(tick, BufferManagerOperationType::REPIN, db, set, pageNum, 0, 0);

  // log the timeline
  logTimeline(tick);
}

void PDBBufferManagerDebugFrontend::logFreeAnonymousPage(uint64_t pageNumber) {

  // lock the timeline file
  std::unique_lock<std::mutex> lck(m);

  // increment the debug tick
  uint64_t tick = debugTick++;

  // log the operation
  logOperation(tick, BufferManagerOperationType::FREE, "", "", pageNumber, 0, 0);

  // log the timeline
  logTimeline(tick);
}

void PDBBufferManagerDebugFrontend::logDownToZeroReferences(const PDBSetPtr &setPtr, size_t pageNum) {

  // lock the timeline file
  std::unique_lock<std::mutex> lck(m);

  // increment the debug tick
  uint64_t tick = debugTick++;

  // log the operation
  logOperation(tick, BufferManagerOperationType::FREE, setPtr->getDBName(), setPtr->getSetName(), pageNum, 0, 0);

  // log the timeline
  logTimeline(tick);
}

void PDBBufferManagerDebugFrontend::logClearSet(const PDBSetPtr &set) {

  // lock the timeline file
  std::unique_lock<std::mutex> lck(m);

  // increment the debug tick
  uint64_t tick = debugTick++;

  // log the operation
  logOperation(tick, BufferManagerOperationType::CLEAR, set->getDBName(), set->getSetName(), 0, 0, 0);

  // log the timeline
  logTimeline(tick);
}

void PDBBufferManagerDebugFrontend::logForward(const Handle<pdb::BufForwardPageRequest> &request) {

  // lock the timeline file
  std::unique_lock<std::mutex> lck(m);

  // increment the debug tick
  uint64_t tick = debugTick++;

  auto db = request->isAnonymous ? "" : request->dbName;
  auto set = request->isAnonymous ? "" : request->setName;

  // log the operation
  logOperation(tick, BufferManagerOperationType::FORWARD, db, set, request->pageNum, 0, request->currentID);

  // log the timeline
  logTimeline(tick);
}

void PDBBufferManagerDebugFrontend::logOperation(uint64_t timestamp,
                                                 PDBBufferManagerDebugFrontend::BufferManagerOperationType operation,
                                                 const string &dbName,
                                                 const string &setName,
                                                 uint64_t pageNumber,
                                                 uint64_t value,
                                                 uint64_t backendID) {

  // store the track if needed
  auto trace = bs::stacktrace();

  // preallocate the memory
  std::vector<size_t> address;
  address.reserve(trace.size());

  // store the trace addresses
  for (bs::frame frame: trace) {
    address.emplace_back((size_t) frame.address());
  }

  // store the stack trace
  auto it = stackTraces.find(address);
  if(it == stackTraces.end()) {

    // insert the stack trace since we don't have it
    it = stackTraces.insert(it, std::make_pair(address, stackTraces.size()));

    // log the ID
    write(debugSymbolTableFile, &it->second, sizeof(it->second));

    // log the size of the bytes of the trace
    uint64_t tmp = it->first.size();
    write(debugSymbolTableFile, &tmp, sizeof(tmp));

    // log the address
    write(debugSymbolTableFile, address.data(), sizeof(size_t) * address.size());

    // convert trace to string
    std::stringstream ss;
    ss << trace;
    std::string s = ss.str();

    // write out the size of the string
    auto size = s.size();
    write(debugSymbolTableFile, &size, sizeof(size));

    // write out the string
    write(debugSymbolTableFile, s.c_str(), sizeof(char) * size);
  }

  // write out the timestamp
  write(stackTracesTableFile, &timestamp, sizeof(timestamp));

  // write out the trace id
  write(stackTracesTableFile, &it->second, sizeof(it->second));

  // write out the operation
  uint64_t tmp;
  switch (operation) {
    case BufferManagerOperationType::GET_PAGE : { tmp = 0; break; }
    case BufferManagerOperationType::FREEZE : { tmp = 1; break; }
    case BufferManagerOperationType::UNPIN : { tmp = 2; break; }
    case BufferManagerOperationType::REPIN : { tmp = 3; break; }
    case BufferManagerOperationType::FREE : { tmp = 4; break; }
    case BufferManagerOperationType::CLEAR : { tmp = 5; break; }
    case BufferManagerOperationType::FORWARD : { tmp = 6; break; }
    case BufferManagerOperationType::HANDLE_PIN_PAGE : { tmp = 7; break; }
    case BufferManagerOperationType::HANDLE_FREEZE_SIZE : { tmp = 8; break; }
    case BufferManagerOperationType::HANDLE_RETURN_PAGE : { tmp = 9; break; }
    case BufferManagerOperationType::HANDLE_GET_PAGE : { tmp = 10; break; }
    case BufferManagerOperationType::HANDLE_UNPIN_PAGE : { tmp = 11; break; }
  }
  write(stackTracesTableFile, &tmp, sizeof(tmp));

  // write out the database name
  tmp = dbName.size();
  write(stackTracesTableFile, &tmp, sizeof(tmp));

  // write out the string
  write(stackTracesTableFile, dbName.c_str(), sizeof(char) * dbName.size());

  // write out the set name
  tmp = setName.size();
  write(stackTracesTableFile, &tmp, sizeof(tmp));

  // write out the string
  write(stackTracesTableFile, setName.c_str(), sizeof(char) * setName.size());

  // write out the page number
  write(stackTracesTableFile, &pageNumber, sizeof(pageNumber));

  // write out the special value
  write(stackTracesTableFile, &value, sizeof(value));

  // write out the backend index
  write(stackTracesTableFile, &backendID, sizeof(backendID));
}

void PDBBufferManagerDebugFrontend::registerHandlers(PDBServer &forMe) {
  forMe.registerHandler(BufGetPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufGetPageRequest>>(
          [&](Handle<BufGetPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

            // call the method to handle it
            auto ret = handleGetPageRequest(request, sendUsingMe);

            {
              // lock the buffer manager to avoid concurrency issues
              std::unique_lock<std::mutex> bufferLock(PDBBufferManagerImpl::m);

              // lock the timeline file
              std::unique_lock<std::mutex> lck(m);

              // increment the debug tick
              uint64_t tick = debugTick++;

              // get these
              std::string db = request->dbName;
              std::string set = request->setName;

              // log the operation
              logOperation(tick, BufferManagerOperationType::HANDLE_GET_PAGE, db, set, request->pageNumber, 0, request->currentID);

              // log the timeline
              logTimeline(tick);
            }

            // return the result
            return ret;
          }));

  forMe.registerHandler(BufGetAnonymousPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufGetAnonymousPageRequest>>(
          [&](Handle<BufGetAnonymousPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

            // call the method to handle it
            auto ret =  handleGetAnonymousPageRequest(request, sendUsingMe);

            {
              // lock the buffer manager to avoid concurrency issues
              std::unique_lock<std::mutex> bufferLock(PDBBufferManagerImpl::m);

              // lock the timeline file
              std::unique_lock<std::mutex> lck(m);

              // increment the debug tick
              uint64_t tick = debugTick++;

              // log the operation
              logOperation(tick, BufferManagerOperationType::HANDLE_GET_PAGE, "", "", 0, request->size, request->currentID);

              // log the timeline
              logTimeline(tick);
            }

            // return the result
            return ret;
          }));

  forMe.registerHandler(BufReturnPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufReturnPageRequest>>(
          [&](Handle<BufReturnPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

            // call the method to handle it
            auto ret =  handleReturnPageRequest(request, sendUsingMe);

            {
              // lock the buffer manager to avoid concurrency issues
              std::unique_lock<std::mutex> bufferLock(PDBBufferManagerImpl::m);

              // lock the timeline file
              std::unique_lock<std::mutex> lck(m);

              // increment the debug tick
              uint64_t tick = debugTick++;

              // log the operation
              logOperation(tick, BufferManagerOperationType::HANDLE_RETURN_PAGE, request->databaseName, request->setName, request->pageNumber, 0, request->currentID);

              // log the timeline
              logTimeline(tick);
            }

            // return the result
            return ret;
          }));

  forMe.registerHandler(BufReturnAnonPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufReturnAnonPageRequest>>(
          [&](Handle<BufReturnAnonPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

            // call the method to handle it
            auto ret =  handleReturnAnonPageRequest(request, sendUsingMe);

            {
              // lock the buffer manager to avoid concurrency issues
              std::unique_lock<std::mutex> bufferLock(PDBBufferManagerImpl::m);

              // lock the timeline file
              std::unique_lock<std::mutex> lck(m);

              // increment the debug tick
              uint64_t tick = debugTick++;

              // log the operation
              logOperation(tick, BufferManagerOperationType::HANDLE_RETURN_PAGE, "", "", request->pageNumber, 0, request->currentID);

              // log the timeline
              logTimeline(tick);
            }

            // return the result
            return ret;
          }));

  forMe.registerHandler(BufFreezeSizeRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufFreezeSizeRequest>>(
          [&](Handle<BufFreezeSizeRequest> request, PDBCommunicatorPtr sendUsingMe) {

            // call the method to handle it
            auto ret =  handleFreezeSizeRequest(request, sendUsingMe);

            {
              // lock the buffer manager to avoid concurrency issues
              std::unique_lock<std::mutex> bufferLock(PDBBufferManagerImpl::m);

              // lock the timeline file
              std::unique_lock<std::mutex> lck(m);

              // increment the debug tick
              uint64_t tick = debugTick++;

              // grab the database and set
              std::string db = request->isAnonymous ? "" : *request->databaseName;
              std::string set = request->isAnonymous ? "" : *request->setName;

              // log the operation
              logOperation(tick, BufferManagerOperationType::HANDLE_FREEZE_SIZE, db, set, request->pageNumber, 0, request->currentID);

              // log the timeline
              logTimeline(tick);
            }

            // return the result
            return ret;
          }));

  forMe.registerHandler(BufPinPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufPinPageRequest>>(
          [&](Handle<BufPinPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

            // call the method to handle it
            auto ret =  handlePinPageRequest(request, sendUsingMe);

            {
              // lock the buffer manager to avoid concurrency issues
              std::unique_lock<std::mutex> bufferLock(PDBBufferManagerImpl::m);

              // lock the timeline file
              std::unique_lock<std::mutex> lck(m);

              // increment the debug tick
              uint64_t tick = debugTick++;

              // grab the database and set
              std::string db = request->isAnonymous ? "" : *request->databaseName;
              std::string set = request->isAnonymous ? "" : *request->setName;

              // log the operation
              logOperation(tick, BufferManagerOperationType::HANDLE_PIN_PAGE, db, set, request->pageNumber, 0,  request->currentID);

              // log the timeline
              logTimeline(tick);
            }

            // return the result
            return ret;
      }));

  forMe.registerHandler(BufUnpinPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<BufUnpinPageRequest>>(
          [&](Handle<BufUnpinPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

          // call the method to handle it
          auto ret =  handleUnpinPageRequest(request, sendUsingMe);
          {
            // lock the buffer manager to avoid concurrency issues
            std::unique_lock<std::mutex> bufferLock(PDBBufferManagerImpl::m);

            // lock the timeline file
            std::unique_lock<std::mutex> lck(m);

            // increment the debug tick
            uint64_t tick = debugTick++;

            // grab the database and set
            std::string db = request->isAnonymous ? "" : *request->databaseName;
            std::string set = request->isAnonymous ? "" : *request->setName;

            // log the operation
            logOperation(tick, BufferManagerOperationType::HANDLE_UNPIN_PAGE, db, set, request->pageNumber, 0, request->currentID);

            // log the timeline
            logTimeline(tick);
          }

          // return the result
          return ret;
      }));
}

PDBBufferManagerInterfacePtr PDBBufferManagerDebugFrontend::getBackEnd() {

  // init the backend storage manager with the shared memory
  return std::make_shared<PDBBufferManagerDebugBackEnd>(sharedMemory, storageLoc);
}

}

#endif