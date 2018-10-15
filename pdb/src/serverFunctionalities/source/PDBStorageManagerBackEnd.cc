//
// Created by dimitrije on 10/12/18.
//

#include <PDBStorageManagerBackEnd.h>
#include <SimpleRequest.h>
#include <StoGetPageRequest.h>
#include <SimpleRequestResult.h>

pdb::PDBStorageManagerBackEnd::PDBStorageManagerBackEnd(const PDBSharedMemory &sharedMemory)
    : sharedMemory(sharedMemory) {

  // register the shared employee class
  myLogger = make_shared<pdb::PDBLogger>("clientLog");
}

pdb::PDBPageHandle pdb::PDBStorageManagerBackEnd::getPage(pdb::PDBSetPtr whichSet, uint64_t i) {

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // somewhere to put the message.
  std::string errMsg;

  // make a request
  auto res = simpleRequest<StoGetPageRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {
        if (result != nullptr) {
          if (!result->getRes().first) {
            errMsg = "Error registering node metadata: " + result->getRes().second;
            myLogger->error("Error registering node metadata: " + result->getRes().second);
            return false;
          }
          return true;
        }
        errMsg = "Error registering node metadata in the catalog";
        return false;
      },
      whichSet->getSetName(), whichSet->getDBName(), i);

  return pdb::PDBPageHandle();
}

pdb::PDBPageHandle pdb::PDBStorageManagerBackEnd::getPage() {
  return pdb::PDBPageHandle();
}

pdb::PDBPageHandle pdb::PDBStorageManagerBackEnd::getPage(size_t minBytes) {
  return pdb::PDBPageHandle();
}

size_t pdb::PDBStorageManagerBackEnd::getPageSize() {
  return 0;
}

void pdb::PDBStorageManagerBackEnd::registerHandlers(pdb::PDBServer &forMe) {

}

void pdb::PDBStorageManagerBackEnd::freeAnonymousPage(pdb::PDBPagePtr me) {

}

void pdb::PDBStorageManagerBackEnd::downToZeroReferences(pdb::PDBPagePtr me) {

}

void pdb::PDBStorageManagerBackEnd::freezeSize(pdb::PDBPagePtr me, size_t numBytes) {

}

void pdb::PDBStorageManagerBackEnd::unpin(pdb::PDBPagePtr me) {

}

void pdb::PDBStorageManagerBackEnd::repin(pdb::PDBPagePtr me) {

}
