#ifdef DEBUG_BUFFER_MANAGER

#include "PDBBufferManagerDebugBackEnd.h"
#include <fcntl.h>
#include <boost/stacktrace.hpp>

namespace pdb {

PDBBufferManagerInterface* PDBBufferManagerDebugBackendFactory::instance = nullptr;

namespace bs = boost::stacktrace;

pdb::PDBBufferManagerDebugBackEnd::PDBBufferManagerDebugBackEnd(const PDBSharedMemory &sharedMemory,
                                                                const std::string &storageLocIn) : PDBBufferManagerBackEnd(sharedMemory) {

  // set me so it can log
  PDBBufferManagerDebugBackendFactory::instance = this;

  // open symbol file
  debugSymbolTableFile = open((storageLocIn + "/debugSymbols.bs").c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);

  // check if we actually opened the file
  if (debugSymbolTableFile == 0) {
    exit(-1);
  }

  // open stackTraces file
  stackTracesTableFile = open((storageLocIn + "/stackTraces.bst").c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);

  // check if we actually opened the file
  if (stackTracesTableFile == 0) {
    exit(-1);
  }
}

void PDBBufferManagerDebugBackEnd::logGetPage(const PDBSetPtr &whichSet, uint64_t i, uint64_t timestamp) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lck(m);

  // log the operation
  logOperation(timestamp, BufferManagerOperationType::GET_PAGE, whichSet->getDBName(), whichSet->getSetName(), i, 0);
}

void PDBBufferManagerDebugBackEnd::logGetPage(size_t minBytes, uint64_t pageNumber, uint64_t timestamp) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lck(m);

  // log the operation
  logOperation(timestamp, BufferManagerOperationType::GET_PAGE, "", "", pageNumber, minBytes);
}

void PDBBufferManagerDebugBackEnd::logFreezeSize(const PDBSetPtr &setPtr, size_t pageNum, size_t numBytes, uint64_t timestamp) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lck(m);

  // get the database and set name
  std::string db = setPtr == nullptr ? "" : setPtr->getDBName();
  std::string set = setPtr == nullptr ? "" : setPtr->getSetName();

  // log the freeze operation
  logOperation(timestamp, BufferManagerOperationType::FREEZE, db, set, pageNum, numBytes);

}

void PDBBufferManagerDebugBackEnd::logUnpin(const PDBSetPtr &setPtr, size_t pageNum, uint64_t timestamp) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lck(m);

  // get the database and set name
  std::string db = setPtr == nullptr ? "" : setPtr->getDBName();
  std::string set = setPtr == nullptr ? "" : setPtr->getSetName();

  // log the operation
  logOperation(timestamp, BufferManagerOperationType::UNPIN, db, set, pageNum, 0);
}

void PDBBufferManagerDebugBackEnd::logRepin(const PDBSetPtr &setPtr, size_t pageNum, uint64_t timestamp) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lck(m);

  // get the database and set name
  std::string db = setPtr == nullptr ? "" : setPtr->getDBName();
  std::string set = setPtr == nullptr ? "" : setPtr->getSetName();

  // log the operation
  logOperation(timestamp, BufferManagerOperationType::REPIN, db, set, pageNum, 0);
}

void PDBBufferManagerDebugBackEnd::logFreeAnonymousPage(uint64_t pageNumber, uint64_t timestamp) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lck(m);

  // log the operation
  logOperation(timestamp, BufferManagerOperationType::FREE, "", "", pageNumber, 0);
}

void PDBBufferManagerDebugBackEnd::logDownToZeroReferences(const PDBSetPtr &setPtr, size_t pageNum, uint64_t timestamp) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lck(m);

  // log the operation
  logOperation(timestamp, BufferManagerOperationType::FREE, setPtr->getDBName(), setPtr->getSetName(), pageNum, 0);
}

void PDBBufferManagerDebugBackEnd::logClearSet(const PDBSetPtr &set, uint64_t timestamp) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lck(m);

  // log the operation
  logOperation(timestamp, BufferManagerOperationType::CLEAR, set->getDBName(), set->getSetName(), 0, 0);
}

void PDBBufferManagerDebugBackEnd::logExpect(const Handle<BufForwardPageRequest> &result) {

  // lock the buffer manager
  std::unique_lock<std::mutex> lck(m);

  // figure out the set name and database name
  auto dbName = result->isAnonymous ? "" : result->dbName;
  auto setName = result->isAnonymous ? "" : result->setName;

  // log the operation
  logOperation(std::numeric_limits<uint64_t>::max() - result->currentID, BufferManagerOperationType::EXPECT, dbName, setName, result->pageNum, 0);
}

void PDBBufferManagerDebugBackEnd::logOperation(uint64_t timestamp,
                                                PDBBufferManagerDebugBackEnd::BufferManagerOperationType operation,
                                                const string &dbName,
                                                const string &setName,
                                                uint64_t pageNumber,
                                                uint64_t value) {

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
    case BufferManagerOperationType::EXPECT : { tmp = 6; break; }
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
}

}

#endif