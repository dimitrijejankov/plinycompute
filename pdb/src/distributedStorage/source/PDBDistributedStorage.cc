/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#include "PDBDistributedStorage.h"
#include <snappy.h>
#include <HeapRequestHandler.h>
#include <DisAddData.h>
#include <DisClearSet.h>
#include <DisRemoveSet.h>
#include <BufGetPageRequest.h>
#include <PDBBufferManagerInterface.h>
#include <PDBDispatchRandomPolicy.h>
#include "PDBCatalogClient.h"
#include <boost/filesystem/path.hpp>
#include <PDBDistributedStorage.h>
#include <fstream>
#include <boost/filesystem/operations.hpp>

namespace pdb {

namespace fs = boost::filesystem;


void PDBDistributedStorage::init() {

  // init the policy
  policy = std::make_shared<PDBDispatchRandomPolicy>();

  // init the class
  logger = make_shared<pdb::PDBLogger>((fs::path(getConfiguration()->rootDirectory) / "logs").string(), "PDBDistributedStorage.log");
}

void PDBDistributedStorage::registerHandlers(PDBServer &forMe) {

forMe.registerHandler(
    StoGetNextPageRequest_TYPEID,
    make_shared<pdb::HeapRequestHandler<pdb::StoGetNextPageRequest>>(
        [&](Handle<pdb::StoGetNextPageRequest> request, PDBCommunicatorPtr sendUsingMe) {
          return handleGetNextPage<PDBCommunicator, RequestFactory>(request, sendUsingMe);
        }));

forMe.registerHandler(
    DisAddData_TYPEID,
    make_shared<HeapRequestHandler<pdb::DisAddData>>(
        [&](Handle<pdb::DisAddData> request, PDBCommunicatorPtr sendUsingMe) {
          return handleAddData<PDBCommunicator, RequestFactory>(request, sendUsingMe);
    }));

forMe.registerHandler(
    DisClearSet_TYPEID,
    make_shared<HeapRequestHandler<pdb::DisClearSet>>(
        [&](Handle<pdb::DisClearSet> request, PDBCommunicatorPtr sendUsingMe) {
          return handleClearSet<PDBCommunicator, RequestFactory>(request, sendUsingMe);
    }));

  forMe.registerHandler(
    DisRemoveSet_TYPEID,
    make_shared<HeapRequestHandler<pdb::DisRemoveSet>>(
        [&](Handle<pdb::DisRemoveSet> request, PDBCommunicatorPtr sendUsingMe) {
          return handleRemoveSet<PDBCommunicator, RequestFactory>(request, sendUsingMe);
    }));
}

pdb::PDBDistributedStorageSetLockPtr pdb::PDBDistributedStorage::useSet(const std::string &dbName,
                                                                        const std::string &setName,
                                                                        pdb::PDBDistributedStorageSetState stateRequested) {

  PDBDistributedStorageSetLockPtr setLock;

  // lock the structure
  std::unique_lock<std::mutex> lck{setInUseLck};

  // wait until we can use the set
  cv.wait(lck, [&] {

    // try to use the set
    setLock = tryUsingSet(dbName, setName, stateRequested, lck);

    // return if is granted, stop waiting
    return setLock->isGranted();
  });

  // return the lock
  return setLock;
}

PDBDistributedStorageSetLockPtr pdb::PDBDistributedStorage::tryUsingSet(const std::string &dbName,
                                                                        const std::string &setName,
                                                                        PDBDistributedStorageSetState stateRequested) {

  // lock the structure
  std::unique_lock<std::mutex> lck{setInUseLck};

  // try to use the set
  return tryUsingSet(dbName, setName, stateRequested, lck);
}

PDBDistributedStorageSetLockPtr pdb::PDBDistributedStorage::tryUsingSet(const std::string &dbName,
                                                                        const std::string &setName,
                                                                        PDBDistributedStorageSetState stateRequested,
                                                                        std::unique_lock<std::mutex> &lck) {



  // grab a ptr to the distributed storage
  auto distStorage = getFunctionalityPtr<pdb::PDBDistributedStorage>();

  // get the
  auto &isInUse = setStates[std::make_pair(dbName, setName)];

  switch (stateRequested) {
    case PDBDistributedStorageSetState::NONE: {
      throw std::runtime_error("You can not request a request of type NONE!");
    }
    case PDBDistributedStorageSetState::WRITE_READ_DATA : {
      throw std::runtime_error("You can not request a request to both read and write (but they can happen at the same time)!");
    }
    case PDBDistributedStorageSetState::WRITING_DATA: {

      // check if we can grant it
      if(isInUse.state == PDBDistributedStorageSetState::NONE ||
         isInUse.state == PDBDistributedStorageSetState::WRITING_DATA ||
         isInUse.state == PDBDistributedStorageSetState::WRITE_READ_DATA) {

        // update the state
        isInUse.numWriters++;

        // update the state if we need
        if(isInUse.state == PDBDistributedStorageSetState::NONE) {
          isInUse.state = PDBDistributedStorageSetState::WRITING_DATA;
        }
        else if(isInUse.state == PDBDistributedStorageSetState::READING_DATA) {
          isInUse.state = PDBDistributedStorageSetState::WRITE_READ_DATA;
        }

        // return
        return std::make_shared<PDBDistributedStorageSetLock>(dbName, setName, PDBDistributedStorageSetState::WRITING_DATA, distStorage);
      }

      // failed to do it
      return std::make_shared<PDBDistributedStorageSetLock>(dbName, setName, PDBDistributedStorageSetState::NONE, distStorage);
    }
    case PDBDistributedStorageSetState::READING_DATA: {

      // check if we can grant it
      if(isInUse.state == PDBDistributedStorageSetState::NONE ||
         isInUse.state == PDBDistributedStorageSetState::READING_DATA ||
         isInUse.state == PDBDistributedStorageSetState::WRITE_READ_DATA) {

        // update the state
        isInUse.numReaders++;

        // update the state if we need
        if(isInUse.state == PDBDistributedStorageSetState::NONE) {
          isInUse.state = PDBDistributedStorageSetState::READING_DATA;
        }
        else if(isInUse.state == PDBDistributedStorageSetState::WRITING_DATA) {
          isInUse.state = PDBDistributedStorageSetState::WRITE_READ_DATA;
        }

        // return
        return std::make_shared<PDBDistributedStorageSetLock>(dbName, setName, PDBDistributedStorageSetState::READING_DATA, distStorage);
      }

      // failed to do it
      return std::make_shared<PDBDistributedStorageSetLock>(dbName, setName, PDBDistributedStorageSetState::NONE, distStorage);
    }
    case PDBDistributedStorageSetState::CLEARING_DATA : {

      // check if we can grant it
      if(isInUse.state == PDBDistributedStorageSetState::NONE) {
        isInUse.state =  PDBDistributedStorageSetState::CLEARING_DATA;

        // return
        return std::make_shared<PDBDistributedStorageSetLock>(dbName, setName, PDBDistributedStorageSetState::CLEARING_DATA, distStorage);
      }

      // failed to do it
      return std::make_shared<PDBDistributedStorageSetLock>(dbName, setName, PDBDistributedStorageSetState::NONE, distStorage);
    }
  }
}

void pdb::PDBDistributedStorage::finishUsingSet(const std::string &dbName, const std::string &setName, PDBDistributedStorageSetState stateRequested) {

  // lock the structure
  std::unique_lock<std::mutex> lck{setInUseLck};

  // get the
  auto &isInUse = setStates[std::make_pair(dbName, setName)];

  if(stateRequested == PDBDistributedStorageSetState::WRITING_DATA &&
     (isInUse.state == PDBDistributedStorageSetState::WRITING_DATA || isInUse.state == PDBDistributedStorageSetState::WRITE_READ_DATA)) {

    // decrement the number of writers
    assert(isInUse.numWriters > 0);
    isInUse.numWriters--;

    // check if we are done writing
    if(isInUse.numReaders == 0 && isInUse.numWriters == 0) {

      // set the state back to none
      isInUse.state = PDBDistributedStorageSetState::NONE;
    }
  }
  else if(stateRequested == PDBDistributedStorageSetState::READING_DATA &&
          (isInUse.state == PDBDistributedStorageSetState::READING_DATA || isInUse.state == PDBDistributedStorageSetState::WRITE_READ_DATA)) {

    // decrement the number of readers
    assert(isInUse.numReaders > 0);
    isInUse.numReaders--;

    // check if we are done reading
    if(isInUse.numReaders == 0 && isInUse.numWriters == 0) {

      // set the state back to none
      isInUse.state = PDBDistributedStorageSetState::NONE;
    }
  }
  else if(stateRequested == PDBDistributedStorageSetState::CLEARING_DATA && isInUse.state == PDBDistributedStorageSetState::CLEARING_DATA) {

    // make some checks
    assert(isInUse.numReaders == 0);
    assert(isInUse.numReaders == 0);

    // set the state back to none
    isInUse.state = PDBDistributedStorageSetState::NONE;
  }
  else {

    // this is not supposed to happen throw an exception
    throw std::runtime_error("There is an invalid state this is not supposed to happen!");
  }

  // notify that we have changed stuff
  cv.notify_all();
}

}