//
// Created by dimitrije on 10/12/18.
//

#include <PDBStorageManagerBackEnd.h>
#include <SimpleRequest.h>
#include <StoGetPageRequest.h>
#include <SimpleRequestResult.h>
#include <StoGetPageResult.h>
#include <StoGetAnonymousPageRequest.h>
#include <StoReturnPageRequest.h>
#include <StoReturnAnonPageRequest.h>

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
  auto res = heapRequest<StoGetPageRequest, StoGetPageResult, pdb::PDBPageHandle>(
      myLogger, port, address, nullptr, 1024,
      [&](Handle<StoGetPageResult> result) {

        if (result != nullptr) {

          PDBPagePtr returnVal = make_shared <PDBPage> (*this);
          returnVal->setMe(returnVal);
          returnVal->isAnon = result->isAnonymous;
          returnVal->pinned = true;
          returnVal->dirty = result->isDirty;
          returnVal->pageNum = result->pageNum;
          returnVal->whichSet = std::make_shared<PDBSet>(result->setName, result->dbName);
          returnVal->location.startPos = result->startPos;
          returnVal->location.numBytes = result->numBytes;
          returnVal->bytes = (void*)(((uint64_t) this->sharedMemory.memory) + (uint64_t) result->offset);

          return make_shared <PDBPageHandleBase> (returnVal);
        }

        // set the error since we failed
        errMsg = "Could not get the requested page";

        return (pdb::PDBPageHandle) nullptr;
      },
      whichSet->getSetName(), whichSet->getDBName(), i);

  // return the page
  return std::move(res);
}

pdb::PDBPageHandle pdb::PDBStorageManagerBackEnd::getPage() {
  return getPage(getConfiguration()->pageSize);
}

pdb::PDBPageHandle pdb::PDBStorageManagerBackEnd::getPage(size_t minBytes) {

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // somewhere to put the message.
  std::string errMsg;

  // make a request
  auto res = heapRequest<StoGetAnonymousPageRequest, StoGetPageResult, pdb::PDBPageHandle>(
      myLogger, port, address, nullptr, 1024,
      [&](Handle<StoGetPageResult> result) {

        if (result != nullptr) {

          PDBPagePtr returnVal = make_shared <PDBPage> (*this);
          returnVal->setMe(returnVal);
          returnVal->isAnon = result->isAnonymous;
          returnVal->pinned = true;
          returnVal->dirty = result->isDirty;
          returnVal->pageNum = result->pageNum;
          returnVal->location.startPos = result->startPos;
          returnVal->location.numBytes = result->numBytes;
          returnVal->bytes = (char*) this->sharedMemory.memory + result->offset;

          // this an anonymous page if it is not set the database and set name
          if(!result->isAnonymous) {
            returnVal->whichSet = std::make_shared<PDBSet>(result->setName, result->dbName);
          }

          return make_shared <PDBPageHandleBase> (returnVal);
        }

        // set the error since we failed
        errMsg = "Could not get the requested page";

        return (pdb::PDBPageHandle) nullptr;
      },
      minBytes);

  // return the page
  return std::move(res);
}

size_t pdb::PDBStorageManagerBackEnd::getPageSize() {
  return getConfiguration()->pageSize;
}

void pdb::PDBStorageManagerBackEnd::freeAnonymousPage(pdb::PDBPagePtr me) {

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // somewhere to put the message.
  std::string errMsg;

  // make a request
  auto res = heapRequest<StoReturnAnonPageRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {

        // return the result
        if (result != nullptr && result->getRes().first) {
          return true;
        }

        // set the error since we failed
        errMsg = "Could not return the requested page";

        return false;
      }, me->pageNum);


  // did we succeed in returning the page
  if(!res) {

    /// TODO figure out what to do in this situation
    // ok something is wrong kill the backend...
    exit(-1);
  }
}

void pdb::PDBStorageManagerBackEnd::downToZeroReferences(pdb::PDBPagePtr me) {

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // somewhere to put the message.
  std::string errMsg;

  // make a request
  auto res = heapRequest<StoReturnPageRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {

        // return the result
        if (result != nullptr && result->getRes().first) {
          return true;
        }

        // set the error since we failed
        errMsg = "Could not return the requested page";

        return false;
      },
      me->whichSet->getSetName(), me->whichSet->getDBName(), me->pageNum);

  // did we succeed in returning the page
  if(!res) {

    /// TODO figure out what to do in this situation
    // ok something is wrong kill the backend...
    exit(-1);
  }
}

void pdb::PDBStorageManagerBackEnd::freezeSize(pdb::PDBPagePtr me, size_t numBytes) {

}

void pdb::PDBStorageManagerBackEnd::unpin(pdb::PDBPagePtr me) {

}

void pdb::PDBStorageManagerBackEnd::repin(pdb::PDBPagePtr me) {

}

void pdb::PDBStorageManagerBackEnd::registerHandlers(pdb::PDBServer &forMe) {

}


