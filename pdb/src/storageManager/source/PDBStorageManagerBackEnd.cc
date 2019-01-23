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
#include <StoFreezeSizeRequest.h>
#include <StoUnpinPageRequest.h>
#include <StoPinPageRequest.h>
#include <StoPinPageResult.h>
#include <mutex>
#include <PDBPageBackend.h>

pdb::PDBStorageManagerBackEnd::PDBStorageManagerBackEnd(const PDBSharedMemory &sharedMemory) : sharedMemory(sharedMemory) {

  // make a logger
  myLogger = make_shared<pdb::PDBLogger>("storageLog");
}

pdb::PDBPageHandle pdb::PDBStorageManagerBackEnd::getPage(pdb::PDBSetPtr whichSet, uint64_t i) {

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // somewhere to put the message.
  std::string errMsg;

  // grab a page handle to this
  auto pageHandle = getBackendPage(whichSet, i);

  // lock the page
  pageHandle->lock();

  // do we have the page return it!
  if (pageHandle->getBytes() != nullptr) {

    // unlock the page
    pageHandle->unlock();

    // return it
    return pageHandle;
  }

  // make a request
  auto res = heapRequest<StoGetPageRequest, StoGetPageResult, pdb::PDBPageHandle>(
      myLogger, port, address, nullptr, 1024,
      [&](Handle<StoGetPageResult> result) {

        if (result != nullptr) {

          // fill in the stuff
          PDBPagePtr returnVal = pageHandle->page;
          returnVal->isAnon = result->isAnonymous;
          returnVal->pinned = true;
          returnVal->dirty = result->isDirty;
          returnVal->pageNum = result->pageNum;
          returnVal->whichSet = std::make_shared<PDBSet>(result->setName, result->dbName);
          returnVal->location.startPos = result->startPos;
          returnVal->location.numBytes = result->numBytes;
          returnVal->bytes = (void *) (((uint64_t) this->sharedMemory.memory) + (uint64_t) result->offset);

          // unlock the page
          pageHandle->unlock();

          return pageHandle;
        }

        auto page = pageHandle->page;

        safeRemoveBackendPage(page);

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

          PDBPagePtr returnVal = make_shared<PDBPageBackend>(*this);
          returnVal->setMe(returnVal);
          returnVal->isAnon = result->isAnonymous;
          returnVal->pinned = true;
          returnVal->dirty = result->isDirty;
          returnVal->pageNum = result->pageNum;
          returnVal->location.startPos = result->startPos;
          returnVal->location.numBytes = result->numBytes;
          returnVal->bytes = (char *) this->sharedMemory.memory + result->offset;

          // this an anonymous page if it is not set the database and set name
          if (!result->isAnonymous) {
            returnVal->whichSet = std::make_shared<PDBSet>(result->setName, result->dbName);
          }

          return make_shared<PDBPageHandleBase>(returnVal);
        }

        // set the error since we failed
        errMsg = "Could not get the requested page";

        return (pdb::PDBPageHandle) nullptr;
      },
      minBytes);

  // return the page
  return std::move(res);
}

size_t pdb::PDBStorageManagerBackEnd::getMaxPageSize() {
  return getConfiguration()->pageSize;
}

pdb::PDBPageHandle pdb::PDBStorageManagerBackEnd::getBackendPage(pdb::PDBSetPtr whichSet, uint64_t i) {

  // lock
  unique_lock<std::mutex> ul(lck);

  // the key to find the pages
  auto key = std::make_pair(whichSet, i);

  // grab the page
  auto it = allPages.find(key);

  // do we have that page? if not create one
  if (it == allPages.end()) {

    // just make the page
    PDBPagePtr returnVal = make_shared<PDBPageBackend>(*this);
    returnVal->setMe(returnVal);
    returnVal->bytes = nullptr;
    returnVal->whichSet = whichSet;
    returnVal->pageNum = i;

    // store the page
    allPages[key] = returnVal;

    // return
    return make_shared<PDBPageHandleBase>(returnVal);
  }

  // return the page
  return make_shared<PDBPageHandleBase>(it->second);
}

void pdb::PDBStorageManagerBackEnd::safeRemoveBackendPage(const pdb::PDBPagePtr &page) {

  // lock the allPages structure
  unique_lock<mutex> ul;

  // are we the only handle to the page
  if(page->refCount <= 1) {

    // remove the page since we don't need it really
    this->allPages.erase(make_pair(page->whichSet, page->pageNum));
  }
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
  if (!res) {

    // ok something is wrong kill the backend...
    exit(-1);
  }
}

void pdb::PDBStorageManagerBackEnd::downToZeroReferences(pdb::PDBPagePtr me) {

  // this method is always called when the page provided is locked thus no requests to the frontend for this page
  // can be made! @see pdb::PDBPage::decRefCount

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

          // set the bytes to null
          me->bytes = nullptr;

          // remove the backend page
          safeRemoveBackendPage(me);

          // true because we succeeded :D
          return true;
        }

        // set the error since we failed
        errMsg = "Could not return the requested page";

        return false;
      },
      me->whichSet->getSetName(), me->whichSet->getDBName(), me->pageNum, me->isDirty());

  // did we succeed in returning the page
  if (!res) {

    // ok something is wrong kill the backend...
    exit(-1);
  }
}

void pdb::PDBStorageManagerBackEnd::freezeSize(pdb::PDBPagePtr me, size_t numBytes) {

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // somewhere to put the message.
  std::string errMsg;

  // make a request
  auto res = heapRequest<StoFreezeSizeRequest, SimpleRequestResult, bool>(
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
      me->whichSet, me->pageNum, numBytes);

  // did we succeed in returning the page
  if (!res) {

    // ok something is wrong kill the backend...
    exit(-1);
  }
}

void pdb::PDBStorageManagerBackEnd::unpin(pdb::PDBPagePtr me) {

  // lock the page so that if somebody calls repin or getPage again. We can not progress unless
  // we are sure that is not happening.
  me->lock();

  // are we already unpinned if so just return no need to send messages around
  if(me->bytes == nullptr) {

    // unlock
    me->unlock();

    // finish
    return;
  }

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // somewhere to put the message.
  std::string errMsg;

  // make a request
  auto res = heapRequest<StoUnpinPageRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {

        // return the result
        if (result != nullptr && result->getRes().first) {

          // invalidate the page
          me->bytes = nullptr;

          // unlock the page
          me->unlock();

          // so it worked
          return true;
        }

        // set the error since we failed
        errMsg = "Could not return the requested page";

        // yeah we could not
        return false;
      },
      me->whichSet, me->pageNum);

  // did we succeed in returning the page
  if (!res) {

    // ok something is wrong kill the backend...
    exit(-1);
  }
}

void pdb::PDBStorageManagerBackEnd::repin(pdb::PDBPagePtr me) {

  // lock the page so that if somebody calls repin or getPage again. We can not progress unless
  // we are sure that is not happening.
  me->lock();

  // check whether the page is already pinned, if so no need to repin it
  if(me->bytes != nullptr) {

    // unlock
    me->unlock();

    // finish
    return;
  }

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // somewhere to put the message.
  std::string errMsg;

  // make a request
  auto res = heapRequest<StoPinPageRequest, StoPinPageResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<StoPinPageResult> result) {

        // return the result
        if (result != nullptr && result->success) {

          // figure out the pointer for the offset
          me->bytes = (void *) ((uint64_t) this->sharedMemory.memory + (uint64_t) result->offset);

          // unlock the page
          me->unlock();

          // we succeeded
          return true;
        }

        // set the error since we failed
        errMsg = "Could not return the requested page";

        return false;
      },
      me->whichSet, me->pageNum);

  // did we succeed in returning the page
  if (!res) {

    // ok something is wrong kill the backend...
    exit(-1);
  }
}

void pdb::PDBStorageManagerBackEnd::registerHandlers(pdb::PDBServer &forMe) {

}


