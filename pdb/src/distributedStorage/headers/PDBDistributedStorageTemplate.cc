#pragma once

#include <string>
#include <PDBDistributedStorage.h>
#include <StoGetNextPageRequest.h>
#include <StoGetNextPageResult.h>
#include <StoGetPageRequest.h>
#include <StoGetPageResult.h>
#include <PDBBufferManagerInterface.h>
#include <PDBCatalogClient.h>
#include <StoDispatchData.h>
#include <StoClearSetRequest.h>
#include <DisClearSet.h>
#include <DisRemoveSet.h>
#include <GenericWork.h>

template<class Communicator, class Requests>
std::pair<pdb::PDBPageHandle, size_t> pdb::PDBDistributedStorage::requestPage(const PDBCatalogNodePtr &node,
                                                                              const std::string &databaseName,
                                                                              const std::string &setName,
                                                                              uint64_t &page) {

  // the communicator
  PDBCommunicatorPtr comm = make_shared<PDBCommunicator>();
  string errMsg;

  // try multiple times if we fail to connect
  int numRetries = 0;
  while (numRetries <= getConfiguration()->maxRetries) {

    // connect to the server
    if (!comm->connectToInternetServer(logger, node->port, node->address, errMsg)) {

      // log the error
      logger->error(errMsg);
      logger->error(
          "Can not connect to remote server with port=" + std::to_string(node->port) + " and address=" + node->address
              + ");");

      // throw an exception
      throw std::runtime_error(errMsg);
    }

    // we connected
    break;
  }

  // make a block to send the request
  const UseTemporaryAllocationBlock tempBlock{1024};

  // make the request
  Handle<StoGetPageRequest> pageRequest = makeObject<StoGetPageRequest>(databaseName, setName, page);

  // send the object
  if (!comm->sendObject(pageRequest, errMsg)) {

    // yeah something happened
    logger->error(errMsg);
    logger->error("Not able to send request to server.\n");

    // throw an exception
    throw std::runtime_error(errMsg);
  }

  // get the response and process it
  bool success;
  Handle<StoGetPageResult> result = comm->getNextObject<StoGetPageResult>(success, errMsg);

  // did we get a response
  if (result == nullptr) {

    // throw an exception
    throw std::runtime_error(errMsg);
  }

  // do we have a next page
  if (!result->hasPage) {
    return std::make_pair(nullptr, result->size);
  }

  // grab a page
  auto pageHandle = getFunctionalityPtr<PDBBufferManagerInterface>()->getPage(result->size);

  // get the bytes
  auto readSize = Requests::waitForBytes(logger, comm, (char *) pageHandle->getBytes(), result->size, errMsg);

  // did we fail to read it...
  if (readSize == -1) {
    return std::make_pair(nullptr, result->size);
  }

  // set the page number
  page = result->pageNumber;

  // return it
  return std::make_pair(pageHandle, result->size);
}

template<class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBDistributedStorage::handleGetNextPage(const pdb::Handle<pdb::StoGetNextPageRequest> &request,
                                                                           std::shared_ptr<Communicator> &sendUsingMe) {

  /// 1. Check if we are the manager

  if (!this->getConfiguration()->isManager) {

    string error = "Only a manager can serve pages!";

    // make an allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{1024};

    // create an allocation block to hold the response
    pdb::Handle<pdb::StoGetNextPageResult> response = pdb::makeObject<pdb::StoGetNextPageResult>(0, "", 0, false);

    // sends result to requester
    sendUsingMe->sendObject(response, error);

    // this is an issue we simply return false only a manager can serve pages
    return make_pair(false, error);
  }

  /// 2. Wait till we get a lock for reading

  auto setLock = useSet(request->databaseName, request->setName, PDBDistributedStorageSetState::READING_DATA);

  /// 3. Figure out if the node we last accessed is still there

  // sort the nodes
  auto nodes = this->getFunctionality<PDBCatalogClient>().getWorkerNodes();
  sort(nodes.begin(), nodes.end(), [](const pdb::PDBCatalogNodePtr &a, const pdb::PDBCatalogNodePtr &b) { return a->nodeID > b->nodeID; });

  // if this is the first time we are requesting a node the next node is
  // the first node otherwise we try to find the previous node
  auto node = nodes.begin();

  if (!request->isFirst) {

    // try to find the last used node
    node = find_if(nodes.begin(), nodes.end(), [&](pdb::PDBCatalogNodePtr s) { return request->nodeID == s->nodeID; });

    // if the node is not active anymore
    if (node == nodes.end()) {

      string error = "Could not find the specified previous node";

      // make an allocation block
      const pdb::UseTemporaryAllocationBlock tempBlock{1024};

      // create an allocation block to hold the response
      pdb::Handle<pdb::StoGetNextPageResult> response = pdb::makeObject<pdb::StoGetNextPageResult>(0, "", 0, false);

      // sends result to requester
      sendUsingMe->sendObject(response, error);

      // This is an issue we simply return false only a manager can serve pages
      return make_pair(false, error);
    }
  }

  /// 4. Find the next page

  uint64_t currPage = request->page;
  for (auto it = node; it != nodes.end(); ++it) {

    // skip this node
    if (!(*it)->active) {
      continue;
    }

    // grab a page
    auto pageInfo = this->requestPage<Communicator, Requests>(*it, request->databaseName, request->setName, currPage);

    // does it have the page
    if (pageInfo.first != nullptr) {

      // make an allocation block
      const pdb::UseTemporaryAllocationBlock tempBlock{1024};

      // create an allocation block to hold the response
      pdb::Handle<pdb::StoGetNextPageResult> response = pdb::makeObject<pdb::StoGetNextPageResult>(currPage, (*it)->nodeID, pageInfo.second, true);

      // sends result to requester
      string error;
      sendUsingMe->sendObject(response, error);

      // now, send the bytes
      if (!sendUsingMe->sendBytes(pageInfo.first->getBytes(), pageInfo.second, error)) {

        this->logger->error(error);
        this->logger->error("sending page bytes: not able to send data to client.\n");

        // we are done here
        return make_pair(false, string(error));
      }

      // we succeeded
      return make_pair(true, string(""));
    }

    // set the page to 0 since we are grabbing a page from a new node
    currPage = 0;
  }

  // make an allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{1024};

  // create an allocation block to hold the response
  pdb::Handle<pdb::StoGetNextPageResult> response = pdb::makeObject<pdb::StoGetNextPageResult>(0, "", 0, false);

  // sends result to requester
  string error;
  bool success = sendUsingMe->sendObject(response, error);

  // return
  return make_pair(success, error);
}

template<class Communicator>
void pdb::PDBDistributedStorage::respondAddDataWithError(shared_ptr<Communicator> &sendUsingMe, std::string &errMsg) {

  // log the error
  logger->error(errMsg);

  // skip the data part
  sendUsingMe->skipBytes(errMsg);

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};
  Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

  // sends result to requester
  sendUsingMe->sendObject(response, errMsg);
}

template<class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBDistributedStorage::handleAddData(const pdb::Handle<pdb::DisAddData> &request,
                                                                       shared_ptr<Communicator> &sendUsingMe) {

  /// 0. Check if the set exists

  // get the set if exists
  std::string error;
  bool success;
  auto set = getFunctionalityPtr<PDBCatalogClient>()->getSet(request->databaseName, request->setName, error);

  // make sure the set exists
  if (set == nullptr) {

    // make the error string
    std::string errMsg = "The set (" + (string)request->databaseName +  "," + (string)request->setName + ") is does not exist!";

    // respond with error
    respondAddDataWithError(sendUsingMe, errMsg);

    // return the problem
    return make_pair(false, errMsg);
  }

  // lock the set
  auto setLock = tryUsingSet(request->databaseName, request->setName, PDBDistributedStorageSetState::WRITING_DATA);
  if(!setLock->isWriteGranted()) {

    // make the error string
    std::string errMsg = "The set (" + (string)request->databaseName +  "," + (string)request->setName + ") is already in use!";

    // respond with error
    respondAddDataWithError(sendUsingMe, errMsg);

    // return the problem
    return make_pair(false, errMsg);
  }

  // make sure the container type of the set is set to
  if (set->containerType == PDBCatalogSetContainerType::PDB_CATALOG_SET_NO_CONTAINER) {

    // update the container type
    success = getFunctionalityPtr<PDBCatalogClient>()->updateSetContainerType(set->database,
                                                                              set->name,
                                                                              PDBCatalogSetContainerType::PDB_CATALOG_SET_VECTOR_CONTAINER,
                                                                              error);

    // if we failed to update the container type this is an error
    if (!success) {

      // make the error string
      std::string errMsg = "Could not update the container type of the set!";

      // respond with error
      respondAddDataWithError(sendUsingMe, errMsg);

      // return the problem
      return make_pair(false, errMsg);
    }
  }
  // if the existing container type is not a vector
  else if (set->containerType != PDBCatalogSetContainerType::PDB_CATALOG_SET_VECTOR_CONTAINER) {

    // make the error string
    std::string errMsg = "The container type of the set is not a vector. You can only add data to vector sets!";

    // respond with error
    respondAddDataWithError(sendUsingMe, errMsg);

    // return the problem
    return make_pair(false, errMsg);
  }

  /// 1. Receive the bytes onto an anonymous page

  // grab the buffer manager
  auto bufferManager = getFunctionalityPtr<PDBBufferManagerInterface>();

  // figure out how large the compressed payload is
  size_t numBytes = sendUsingMe->getSizeOfNextObject();

  // check if it is larger than a page
  if (bufferManager->getMaxPageSize() < numBytes) {

    // make the error string
    std::string errMsg = "The compressed size is larger than the maximum page size";

    // log the error
    logger->error(errMsg);

    // skip the data part
    sendUsingMe->skipBytes(errMsg);

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};
    Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

    // sends result to requester
    sendUsingMe->sendObject(response, errMsg);
    return make_pair(false, errMsg);
  }

  // grab a page to write this
  auto page = bufferManager->getPage(numBytes);

  // receive bytes
  sendUsingMe->receiveBytes(page->getBytes(), error);

  // check the uncompressed size
  size_t uncompressedSize = 0;
  snappy::GetUncompressedLength((char *) page->getBytes(), numBytes, &uncompressedSize);

  // check the uncompressed size
  if (bufferManager->getMaxPageSize() < uncompressedSize) {

    // make the error string
    std::string errMsg = "The uncompressed size is larger than the maximum page size";

    // log the error
    logger->error(errMsg);

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};
    Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

    // sends result to requester
    sendUsingMe->sendObject(response, errMsg);
    return make_pair(false, errMsg);
  }

  /// 2. Update the set size
  {
    std::string errMsg;
    if (!getFunctionalityPtr<PDBCatalogClient>()->incrementSetSize(request->databaseName,
                                                                   request->setName,
                                                                   uncompressedSize,
                                                                   errMsg)) {

      // create an allocation block to hold the response
      const UseTemporaryAllocationBlock tempBlock{1024};
      Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

      // sends result to requester
      sendUsingMe->sendObject(response, errMsg);
      return make_pair(false, errMsg);
    }
  }

  /// 3. Figure out on what node to forward the thing

  // grab all active nodes
  const auto nodes = getFunctionality<PDBCatalogClient>().getActiveWorkerNodes();

  // if we have no nodes
  if (nodes.empty()) {

    // make the error string
    std::string errMsg = "There are no nodes where we can dispatch the data to!";

    // log the error
    logger->error(errMsg);

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};
    Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

    // sends result to requester
    sendUsingMe->sendObject(response, errMsg);
    return make_pair(false, errMsg);
  }

  // get the next node
  auto node = policy->getNextNode(request->databaseName, request->setName, nodes);

  // time to send the stuff
  auto ret = RequestFactory::bytesHeapRequest<StoDispatchData, SimpleRequestResult, bool>(
      logger, node->port, node->address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {

        if (result != nullptr && result->getRes().first) {
          return true;
        }

        logger->error("Error sending data: " + result->getRes().second);
        error = "Error sending data: " + result->getRes().second;

        return false;
      },
      (char *) page->getBytes(), numBytes, request->databaseName, request->setName, request->typeName, numBytes);


  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};
  Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(ret, error);

  // sends result to requester
  ret = sendUsingMe->sendObject(response, error) && ret;

  return make_pair(ret, error);
}

template<class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBDistributedStorage::handleClearSet(const pdb::Handle<pdb::DisClearSet> &request,
                                                                        shared_ptr<Communicator> &sendUsingMe) {
  /// 0. Check if the set exists

  // get the set if exists
  std::string error;
  auto set = getFunctionalityPtr<PDBCatalogClient>()->getSet(request->databaseName, request->setName, error);

  // make sure the set exists
  if (set == nullptr) {

    // make the error string
    std::string errMsg =
        "The set requested (" + (std::string) request->databaseName + "," + (std::string) request->setName
            + ") to clear does note exist!";

    // log the error
    logger->error(errMsg);

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};
    Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

    // sends result to requester
    sendUsingMe->sendObject(response, errMsg);
    return make_pair(false, errMsg);
  }

  /// 1. Try to use the set, if it is already in use NACK

  auto setLock = tryUsingSet(request->databaseName, request->setName, PDBDistributedStorageSetState::CLEARING_DATA);
  if(!setLock->isClearGranted()) {

    // make the error string
    std::string errMsg =
        "The set requested (" + (std::string) request->databaseName + "," + (std::string) request->setName
            + ") is in use therefore we can not clear it!";

    // log the error
    logger->error(errMsg);

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};
    Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

    // sends result to requester
    sendUsingMe->sendObject(response, errMsg);
    return make_pair(false, errMsg);
  }

  /// 2. Send concurrently all the worker nodes a request for the storage to clear the set.

  // grab the nodes we want to forward the request to
  auto nodes = getFunctionality<PDBCatalogClient>().getActiveWorkerNodes();

  // success indicator
  atomic_bool success;
  success = true;

  // create the buzzer
  atomic_int counter;
  counter = 0;
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // send a request to clear to each storage
  for (auto i = 0; i < nodes.size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = this->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&, nodeID = i](PDBBuzzerPtr callerBuzzer) {

      // make a request and return the value
      success = RequestFactory::heapRequest<pdb::StoClearSetRequest, SimpleRequestResult, bool>(
          logger, nodes[nodeID]->port, nodes[nodeID]->address, false, 1024, [&](Handle<SimpleRequestResult> result) {

            // check if we got a ACK
            return result != nullptr && result->getRes().first;
          }, request->databaseName, request->setName);

      // signal that the run was successful
      callerBuzzer->buzz(success ? PDBAlarm::WorkAllDone : PDBAlarm::GenericError, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // wait until all the nodes are finished
  while (counter < nodes.size()) {
    tempBuzzer->wait();
  }

  /// 3. Send back the response to the client

  if (!success) {
    error = "Could not completely clear the set";
  }

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};
  Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(success, error);

  // sends result to requester
  sendUsingMe->sendObject(response, error);

  // return
  return make_pair((bool) success, error);
}

template<class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBDistributedStorage::handleRemoveSet(const pdb::Handle<pdb::DisRemoveSet> &request,
                                                                         shared_ptr<Communicator> &sendUsingMe) {
  /// 0. Check if the set exists

  // get the set if exists
  std::string error;
  auto catalogClient = getFunctionalityPtr<PDBCatalogClient>();
  auto set = catalogClient->getSet(request->databaseName, request->setName, error);

  // make sure the set exists
  if (set == nullptr) {

    // make the error string
    std::string errMsg =
        "The set requested (" + (std::string) request->databaseName + "," + (std::string) request->setName
            + ") to remove does note exist!";

    // log the error
    logger->error(errMsg);

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};
    Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

    // sends result to requester
    sendUsingMe->sendObject(response, errMsg);
    return make_pair(false, errMsg);
  }

  /// 1. Try to use the set, if it is already in use NACK

  auto setLock = tryUsingSet(request->databaseName, request->setName, PDBDistributedStorageSetState::CLEARING_DATA);
  if(!setLock->isClearGranted()) {

    // make the error string
    std::string errMsg =
        "The set requested (" + (std::string) request->databaseName + "," + (std::string) request->setName
            + ") is in use therefore we can not remove it!";

    // log the error
    logger->error(errMsg);

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};
    Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

    // sends result to requester
    sendUsingMe->sendObject(response, errMsg);
    return make_pair(false, errMsg);
  }

  /// 2. Remove the set from the catalog
  atomic_bool success;
  success = catalogClient->removeSet(request->databaseName, request->setName, error);
  if(!success) {

    // make the error string
    std::string errMsg =
        "The set requested (" + (std::string) request->databaseName + "," + (std::string) request->setName
            + ") could not be removed from the catalog and therefore we can not remove it!. Error was : " + error;

    // log the error
    logger->error(errMsg);

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};
    Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, errMsg);

    // sends result to requester
    sendUsingMe->sendObject(response, errMsg);
    return make_pair(false, errMsg);
  }

  /// 2. Send concurrently all the worker nodes a request for the storage to clear the set.

  // grab the nodes we want to forward the request to
  auto nodes = getFunctionality<PDBCatalogClient>().getActiveWorkerNodes();

  // success indicator
  success = true;

  // create the buzzer
  atomic_int counter;
  counter = 0;
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // send a request to clear to each storage
  for (auto i = 0; i < nodes.size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = this->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&, nodeID = i](PDBBuzzerPtr callerBuzzer) {

      // make a request and return the value
      success = RequestFactory::heapRequest<pdb::StoClearSetRequest, SimpleRequestResult, bool>(
          logger, nodes[nodeID]->port, nodes[nodeID]->address, false, 1024, [&](Handle<SimpleRequestResult> result) {

            // check if we got a ACK
            return result != nullptr && result->getRes().first;
          }, request->databaseName, request->setName);

      // signal that the run was successful
      callerBuzzer->buzz(success ? PDBAlarm::WorkAllDone : PDBAlarm::GenericError, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // wait until all the nodes are finished
  while (counter < nodes.size()) {
    tempBuzzer->wait();
  }

  /// 3. Send back the response to the client

  if (!success) {
    error = "Could not completely remove the set, failed to clear its data.";
  }

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};
  Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(success, error);

  // sends result to requester
  sendUsingMe->sendObject(response, error);

  // return
  return make_pair((bool) success, error);
}
