#ifndef BE_STORAGE_MGR_CCT
#define BE_STORAGE_MGR_CCT

#include "PDBBufferManagerBackEnd.h"
#include <HeapRequest.h>
#include <BufGetPageRequest.h>
#include <SimpleRequestResult.h>
#include <BufForwardPageRequest.h>
#include <BufGetPageResult.h>
#include <BufGetAnonymousPageRequest.h>
#include <BufReturnPageRequest.h>
#include <BufReturnAnonPageRequest.h>
#include <BufFreezeSizeRequest.h>
#include <BufUnpinPageRequest.h>
#include <BufPinPageRequest.h>
#include <BufPinPageResult.h>
#include <mutex>
#include <BufFreezeRequestResult.h>

namespace pdb {

template <class T>
pdb::PDBBufferManagerBackEnd<T>::PDBBufferManagerBackEnd(const PDBSharedMemory &sharedMemory) : sharedMemory(sharedMemory) {

  // make a logger
  myLogger = make_shared<pdb::PDBLogger>("storageLog");
}

template <class T>
pdb::PDBPageHandle pdb::PDBBufferManagerBackEnd<T>::getPage(pdb::PDBSetPtr whichSet, uint64_t i) {


  /// 1. Go and check if we are the only ones working on the page if we are mark it with a loading status
  PDBPageHandle pageHandle;
  {
    // lock the page
    unique_lock<std::mutex> lock(m);

    // make the key
    pair<PDBSetPtr, long> key = std::make_pair(whichSet, i);

    cv.wait(lock, [&] {

      // find the page
      auto it = allPages.find(key);

      // if it does not exist create it
      if(it == allPages.end()) {

        // make a page
        PDBPagePtr returnVal = make_shared<PDBPage>(*this);
        returnVal->setBytes(nullptr);
        returnVal->setMe(returnVal);
        returnVal->setSet(whichSet);
        returnVal->setPageNum(i);
        returnVal->status = PDB_PAGE_NOT_LOADED;

        // insert the page
        allPages[key] = returnVal;

        return true;
      }

      // grab the status
      auto status = it->second->status;

      // are we working on it if so wait a bit?
      return !(status == PDB_PAGE_LOADING || status == PDB_PAGE_UNLOADING || status == PDB_PAGE_FREEZING);
    });

    auto it = allPages.find(key);

    // make a page handle
    pageHandle = make_shared<PDBPageHandleBase>(it->second);

    // set the status to loading since we are doing stuff with it
    pageHandle->page->status = PDB_PAGE_LOADING;

    /// 2. At this point it is safe to assume we are the only one with access to the page. Now check if it is already
    /// on the backend

    // do we have the page return it!
    if (pageHandle->getBytes() != nullptr) {

      // mark the page as loaded
      pageHandle->page->status = PDB_PAGE_LOADED;

      lock.unlock();

      // notify all threads that the state has changed
      cv.notify_all();

      // return it
      return pageHandle;
    }
  }

  /// 3. Request the page from the frontend since we don't have it...

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // ok we don't have the page loaded, make a request to get it...
  auto res = T::template heapRequest<BufGetPageRequest, BufGetPageResult, pdb::PDBPageHandle>(
      myLogger, port, address, nullptr, 1024,
      [&](Handle<BufGetPageResult> result) {

        if (result != nullptr) {

          {
            // lock the pages
            unique_lock<std::mutex> lock(m);

            // fill in the stuff
            PDBPagePtr returnVal = pageHandle->page;
            returnVal->isAnon = result->isAnonymous;
            returnVal->pinned = true;
            returnVal->dirty = result->isDirty;
            returnVal->pageNum = result->pageNum;
            returnVal->whichSet = std::make_shared<PDBSet>(result->dbName, result->setName);
            returnVal->location.startPos = result->startPos;
            returnVal->location.numBytes = result->numBytes;
            returnVal->bytes = (void *) (((uint64_t) this->sharedMemory.memory) + (uint64_t) result->offset);
            returnVal->status = PDB_PAGE_LOADED;
          }

          // notify all threads that the state has changed
          cv.notify_all();

          // return the page handle
          return pageHandle;
        }

        // set the error since we failed
        if(whichSet != nullptr) {
          myLogger->error("Could not get the requested page (" + whichSet->getDBName() +  ", " + whichSet->getSetName() + ", " + std::to_string(i) + ")");
        }
        else {
          myLogger->error("Could not get the requested anonymous page with the page number " + std::to_string(i) + ")");
        }

        // something strange happened kill the backend!
        exit(-1);

      },
      whichSet, i);

  // return the page
  return std::move(res);
}

template <class T>
pdb::PDBPageHandle pdb::PDBBufferManagerBackEnd<T>::getPage() {
  return getPage(getConfiguration()->pageSize);
}

template <class T>
pdb::PDBPageHandle pdb::PDBBufferManagerBackEnd<T>::getPage(size_t minBytes) {

  if (minBytes > sharedMemory.pageSize) {
    std::cerr << minBytes << " is larger than the system page size of " << sharedMemory.pageSize << "\n";
    return nullptr;
  }

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // somewhere to put the message.
  std::string errMsg;

  /// 1. We simply request the page it is safe since it is a new anonymous page

  // make a request
  auto res = T::template heapRequest<BufGetAnonymousPageRequest, BufGetPageResult, pdb::PDBPageHandle>(
      myLogger, port, address, nullptr, 1024,
      [&](Handle<BufGetPageResult> result) {

        if (result != nullptr) {

          PDBPagePtr returnVal = make_shared<PDBPage>(*this);
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
            returnVal->whichSet = std::make_shared<PDBSet>(result->dbName, result->setName);
          }

          // put in the the all pages
          {
            // lock all pages to add the page there
            unique_lock<std::mutex> lck(m);

            // mark the page as loaded
            returnVal->status = PDB_PAGE_LOADED;

            // insert the page
            allPages[std::make_pair(returnVal->whichSet, returnVal->pageNum)] = returnVal;
          }

          // notify all threads that the state has changed
          cv.notify_all();

          // return the page handle
          return make_shared<PDBPageHandleBase>(returnVal);
        }

        // set the error since we failed
        myLogger->error("Could not get the requested anonymous page of size " + std::to_string(minBytes));

        return (pdb::PDBPageHandle) nullptr;
      },
      minBytes);

  // return the page
  return std::move(res);
}

template<class T>
PDBPageHandle PDBBufferManagerBackEnd<T>::expectPage(std::shared_ptr<PDBCommunicator> &communicator) {

  /// 1. Wait to receive the request for forwarding

  size_t objectSize = communicator->getSizeOfNextObject();

  // check if we did get a response
  if (objectSize == 0) {

    // ok we did not that sucks log what happened
    myLogger->error("we did not get a response.\n");
  }

  // allocate the memory
  std::unique_ptr<char[]> memory(new char[objectSize]);

  // block to get the response
  bool success;
  std::string error;
  Handle<BufForwardPageRequest> result = communicator->template getNextObject<BufForwardPageRequest>(memory.get(), success, error);

  // did we fail
  if (!success) {

    // log the error
    myLogger->error(error);
    myLogger->error("not able to get forward request over the wire.\n");

    // we failed
    return nullptr;
  }

  /// 2. we got the forward do the book keeping

  PDBPageHandle pageHandle;
  {
    // lock the page
    unique_lock<std::mutex> lock(m);

    // make the set
    auto whichSet = make_shared<PDBSet>(result->dbName, result->setName);

    // make the key
    pair<PDBSetPtr, long> key = std::make_pair(whichSet, result->pageNum);

    cv.wait(lock, [&] {

      // find the page
      auto it = allPages.find(key);

      // if it does not exist create it with the info from the forward page
      if(it == allPages.end()) {

        // make a page
        PDBPagePtr returnVal = make_shared<PDBPage>(*this);
        returnVal->setMe(returnVal);
        returnVal->isAnon = result->isAnonymous;
        returnVal->pinned = true;
        returnVal->dirty = result->isDirty;
        returnVal->pageNum = result->pageNum;
        returnVal->whichSet = result->isAnonymous ? nullptr : std::make_shared<PDBSet>(result->dbName, result->setName);
        returnVal->location.startPos = result->startPos;
        returnVal->location.numBytes = result->numBytes;
        returnVal->bytes = (void *) (((uint64_t) this->sharedMemory.memory) + (uint64_t) result->offset);
        returnVal->status = PDB_PAGE_LOADED;

        // log the expect
        logExpect(result);

        // insert the page
        allPages[key] = returnVal;

        return true;
      }

      // grab the status
      auto status = it->second->status;

      // are we working on it if so wait a bit?
      return !(status == PDB_PAGE_LOADING || status == PDB_PAGE_UNLOADING || status == PDB_PAGE_FREEZING);
    });

    // find the page
    auto it = allPages.find(key);

    // make a page handle
    pageHandle = make_shared<PDBPageHandleBase>(it->second);

    /// 3. At this point it is safe to assume we are the only one with access to the page. Now check if it is unpinned

    // is the page maybe not loaded, then fill it up with the forwarding info. Note we are not caring if somebody
    // unpinned the page in the mean time, since this is an obvious usage error
    if (pageHandle->getBytes() == nullptr) {

      // grab the page
      auto &page = pageHandle->page;

      // mark the page as loaded
      page->isAnon = result->isAnonymous;
      page->pinned = true;
      page->dirty = result->isDirty;
      page->pageNum = result->pageNum;
      page->whichSet = result->isAnonymous ? nullptr : std::make_shared<PDBSet>(result->dbName, result->setName);
      page->location.startPos = result->startPos;
      page->location.numBytes = result->numBytes;
      page->bytes = (void *) (((uint64_t) this->sharedMemory.memory) + (uint64_t) result->offset);
      page->status = PDB_PAGE_LOADED;

      lock.unlock();

      // notify all threads that the state has changed
      cv.notify_all();
    }
  }

  /// 4. Send an acknowledgment that we received the page

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // create the response
  Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(true, std::string("Could not find the page to remove!"));

  // sends result to requester
  std::string errMsg;
  bool res = communicator->sendObject(response, errMsg);

  // if we fail
  if(!res) {
    return nullptr;
  }

  return pageHandle;
}

template <class T>
size_t pdb::PDBBufferManagerBackEnd<T>::getMaxPageSize() {
  return getConfiguration()->pageSize;
}

template <class T>
void pdb::PDBBufferManagerBackEnd<T>::freeAnonymousPage(pdb::PDBPagePtr me) {

  /// 1. Since the count of references for the anon page has hit zero we simply remove it from the allPages

  PDBPageHandle pageHandle;
  {
    // lock the pages
    unique_lock<std::mutex> lck(m);

    // remove if from all the pages
    auto key = std::make_pair(std::make_shared<PDBSet>("", ""), me->whichPage());

    // just remove the page
    allPages.erase(key);
  }

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  /// 2. We make a request to return it to the other side

  // make a request
  auto res = T::template heapRequest<BufReturnAnonPageRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {

        // return the result
        if (result != nullptr && result->getRes().first) {
          return true;
        }

        // set the error since we failed
        myLogger->error("Could not return the requested page");

        return false;
      }, me->pageNum, me->isDirty());


  // did we succeed in returning the page
  if (!res) {

    // ok something is wrong kill the backend...
    exit(-1);
  }
}

template <class T>
void pdb::PDBBufferManagerBackEnd<T>::downToZeroReferences(pdb::PDBPagePtr me) {

  /// 1. Wait till we have exclusive access to tha page and then check if the removal is still valid
  PDBPageHandle pageHandle;
  {
    // lock the pages
    unique_lock<std::mutex> lck(m);

    // wait as long as something is happening with the page
    cv.wait(lck, [&] { return !(me->status == PDB_PAGE_LOADING || me->status == PDB_PAGE_UNLOADING || me->status == PDB_PAGE_FREEZING); });

    // check the reference count
    {
      unique_lock<std::mutex> pageLck(me->lk);

      // ok if by some chance we made another reference while we were waiting for the lock
      // do not free it!
      if(me->refCount != 0) {
        return;
      }
    }

    // find the page
    auto it = allPages.find(std::make_pair(me->whichSet, me->whichPage()));

    // was the page removed while we were waiting if so we are done here
    if(it == allPages.end()) {
      return;
    }

    // was the page removed and then replaced by another one while we were waiting
    if(it->second.get() != me.get()) {
      return;
    }

    // mark the page as unloading
    me->status = PDB_PAGE_UNLOADING;
  }

  /// 2. We need to notify the frontend that we are returning this page

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // make a request
  auto res = T::template heapRequest<BufReturnPageRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {

        // return the result
        if (result != nullptr && result->getRes().first) {

          // do the book keeping
          {
            // lock the pages
            unique_lock<std::mutex> lck(m);

            // remove the page
            allPages.erase(std::make_pair(me->whichSet, me->whichPage()));

            // set the bytes to null
            me->bytes = nullptr;

            // mark it as unloaded
            me->status = PDB_PAGE_NOT_LOADED;
          }

          // notify all threads that the state has changed
          cv.notify_all();

          // true because we succeeded :D
          return true;
        }

        // set the error since we failed
        myLogger->error("Could not return the requested page");

        return false;
      },
      me->whichSet->getSetName(), me->whichSet->getDBName(), me->pageNum, me->isDirty());

  // did we succeed in returning the page
  if (!res) {

    // ok something is wrong kill the backend...
    exit(-1);
  }
}

template <class T>
void pdb::PDBBufferManagerBackEnd<T>::freezeSize(pdb::PDBPagePtr me, size_t numBytes) {

  /// 1.  Make sure we are the only ones working on the page
  PDBPageHandle pageHandle;
  {
    // lock the pages
    unique_lock<std::mutex> lck(m);

    // wait as long as something is happening with the page
    cv.wait(lck, [&] { return !(me->status == PDB_PAGE_LOADING || me->status == PDB_PAGE_UNLOADING || me->status == PDB_PAGE_FREEZING); });

    // mark that we are freezing the page
    me->status = PDB_PAGE_FREEZING;
  }

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // somewhere to put the message.
  std::string errMsg;

  /// 2. Make the request to freeze it

  // make a request
  auto res = T::template heapRequest<BufFreezeSizeRequest, BufFreezeRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<BufFreezeRequestResult> result) {

        // return the result
        if (result != nullptr && result->res) {

          {
            // lock the pages
            unique_lock<std::mutex> lck(m);

            // mark the thing as frozen
            me->sizeFrozen = true;
            me->status = PDB_PAGE_LOADED;

            // notify all threads that the state has changed
            cv.notify_all();
          }

          return true;
        }

        // set the error since we failed
        myLogger->error("Could not freeze the page page");

        return false;
      },
      me->whichSet, me->pageNum, numBytes);

  // did we succeed in returning the page
  if (!res) {

    // ok something is wrong kill the backend...
    exit(-1);
  }
}

template <class T>
void pdb::PDBBufferManagerBackEnd<T>::unpin(pdb::PDBPagePtr me) {

  PDBPageHandle pageHandle;
  {
    // lock the pages
    unique_lock<std::mutex> lck(m);

    // wait as long as something is happening with the page
    cv.wait(lck, [&] { return !(me->status == PDB_PAGE_LOADING || me->status == PDB_PAGE_UNLOADING || me->status == PDB_PAGE_FREEZING); });

    // update status
    me->status = PDB_PAGE_UNLOADING;

    // are we already unpinned if so just return no need to send messages around
    if(me->bytes == nullptr) {

      // mark the page as unloaded
      me->status = PDB_PAGE_NOT_LOADED;

      // unlock
      lck.unlock();

      // notify all threads that the state has changed
      cv.notify_all();

      // finish
      return;
    }
  }

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // somewhere to put the message.
  std::string errMsg;

  // make a request
  auto res = T::template heapRequest<BufUnpinPageRequest, SimpleRequestResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {

        // return the result
        if (result != nullptr && result->getRes().first) {

          {
            // lock the page
            unique_lock<std::mutex> lck(m);

            // invalidate the page
            me->bytes = nullptr;
            me->status = PDB_PAGE_NOT_LOADED;
          }

          // notify all threads that the state has changed
          cv.notify_all();

          // so it worked
          return true;
        }

        // set the error since we failed
        errMsg = "Could not return the requested page";

        // yeah we could not
        return false;
      },
      me->whichSet, me->pageNum, me->isDirty());

  // did we succeed in returning the page
  if (!res) {

    // ok something is wrong kill the backend...
    exit(-1);
  }
}

template <class T>
void pdb::PDBBufferManagerBackEnd<T>::repin(pdb::PDBPagePtr me) {

  PDBPageHandle pageHandle;
  {
    // lock the page
    unique_lock<std::mutex> lck(m);

    // wait as long as something is happening with the page
    cv.wait(lck, [&] { return !(me->status == PDB_PAGE_LOADING || me->status == PDB_PAGE_UNLOADING || me->status == PDB_PAGE_FREEZING); });

    // update status
    me->status = PDB_PAGE_LOADING;

    // check whether the page is already pinned, if so no need to repin it
    if(me->bytes != nullptr) {

      // mark as loaded
      me->status = PDB_PAGE_LOADED;

      // finish
      return;
    }
  }

  // grab the address of the frontend
  auto port = getConfiguration()->port;
  auto address = getConfiguration()->address;

  // somewhere to put the message.
  std::string errMsg;

  // make a request
  auto res = T::template heapRequest<BufPinPageRequest, BufPinPageResult, bool>(
      myLogger, port, address, false, 1024,
      [&](Handle<BufPinPageResult> result) {

        // return the result
        if (result != nullptr && result->success) {

          {
            // lock the page
            unique_lock<std::mutex> lck(m);

            // figure out the pointer for the offset and update status
            me->bytes = (void *) ((uint64_t) this->sharedMemory.memory + (uint64_t) result->offset);
            me->status = PDB_PAGE_LOADED;
          }

          // notify all threads that the state has changed
          cv.notify_all();

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

template <class T>
void pdb::PDBBufferManagerBackEnd<T>::registerHandlers(pdb::PDBServer &forMe) {}



}

#endif


