#include "PDBDistributedStorageSetLock.h"
#include "PDBDistributedStorage.h"

#include <string>

pdb::PDBDistributedStorageSetLock::PDBDistributedStorageSetLock(const std::string &dbName,
                                                                const std::string &setName,
                                                                pdb::PDBDistributedStorageSetState state,
                                                                const pdb::PDBDistributedStoragePtr &storage)
    : state(state), storage(storage), dbName(dbName), setName(setName) {}


pdb::PDBDistributedStorageSetLock::~PDBDistributedStorageSetLock() {
  storage->finishUsingSet(dbName, setName, state);
}

bool pdb::PDBDistributedStorageSetLock::isWriteGranted() {
  return state == PDBDistributedStorageSetState::WRITING_DATA;
}

bool pdb::PDBDistributedStorageSetLock::isReadGranted() {
  return state == PDBDistributedStorageSetState::READING_DATA;
}

bool pdb::PDBDistributedStorageSetLock::isClearGranted() {
  return state == PDBDistributedStorageSetState::CLEARING_DATA;
}

bool pdb::PDBDistributedStorageSetLock::isGranted() {
  return state != PDBDistributedStorageSetState::NONE;
}

