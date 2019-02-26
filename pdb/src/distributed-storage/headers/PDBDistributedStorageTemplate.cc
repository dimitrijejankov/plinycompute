//
// Created by dimitrije on 2/26/19.
//

#ifndef PDB_PDBDISTRIBUTEDSTORAGETEMPLATE_CC
#define PDB_PDBDISTRIBUTEDSTORAGETEMPLATE_CC

#include <string>
#include <PDBDistributedStorage.h>
#include <StoGetNextPageRequest.h>
#include <StoGetNextPageResult.h>
#include <StoGetPageRequest.h>
#include <StoGetPageResult.h>
#include <PDBBufferManagerInterface.h>
#include <PDBCatalogClient.h>

template <class Communicator, class Requests>
std::pair<pdb::PDBPageHandle, size_t> pdb::PDBDistributedStorage::requestPage(const PDBCatalogNodePtr& node, const std::string &databaseName, const std::string &setName, uint64_t page) {

  // the communicator
  PDBCommunicatorPtr comm = make_shared<PDBCommunicator>();
  string errMsg;

  // try multiple times if we fail to connect
  int numRetries = 0;
  while (numRetries <= getConfiguration()->maxRetries) {

    // connect to the server
    if (comm->connectToInternetServer(logger, node->port, node->address, errMsg)) {

      // log the error
      logger->error(errMsg);
      logger->error("Can not connect to remote server with port=" + std::to_string(node->port) + " and address=" + node->address + ");");

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
  Handle<StoGetPageResult> result = comm->getNextObject<StoGetPageResult> (success, errMsg);

  // did we get a response
  if(result == nullptr) {

    // throw an exception
    throw std::runtime_error(errMsg);
  }

  // do we have a next page
  if(!result->hasPage) {
    return std::make_pair(nullptr, result->size);
  }

  // grab a page
  auto pageHandle = getFunctionalityPtr<PDBBufferManagerInterface>()->getPage(result->size);

  // get the bytes
  auto readSize = Requests::waitForBytes(logger, comm, (char*) pageHandle->getBytes(), result->size, errMsg);

  // did we fail to read it...
  if(readSize == -1) {
    return std::make_pair(nullptr, result->size);
  }

  // return it
  return std::make_pair(pageHandle, result->size);
}

template <class Communicator, class Requests>
std::pair<bool, std::string> pdb::PDBDistributedStorage::handleGetNextPage(const pdb::Handle<pdb::StoGetNextPageRequest> &request,
                                                                               std::shared_ptr<Communicator>  &sendUsingMe) {

  /// 1. Check if we are the manager

  if(!this->getConfiguration()->isManager) {

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

  /// 2. Figure out if the node we last accessed is still there

  // sort the nodes
  auto nodes = this->getFunctionality<PDBCatalogClient>().getActiveWorkerNodes();
  sort(nodes.begin(), nodes.end(), [](const pdb::PDBCatalogNodePtr & a, const pdb::PDBCatalogNodePtr & b) { return a->nodeID > b->nodeID; });

  // if this is the first time we are requesting a node the next node is
  // the first node otherwise we try to find the previous node
  auto node = nodes.begin();

  if(!request->isFirst) {

    // try to find the last used node
    node = find_if(nodes.begin(), nodes.end(), [&] (pdb::PDBCatalogNodePtr s) { return request->nodeID == s->nodeID; } );

    // if the node is not active anymore
    if(node == nodes.end()) {

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

  /// 3. Find the next page

  uint64_t currPage = request->page;
  for(auto it = node; it != nodes.end(); ++it) {

    // grab a page
    auto pageInfo = this->requestPage<Communicator, Requests>(*it, request->databaseName, request->setName, currPage);

    // does it have the page
    if(pageInfo.first != nullptr) {

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

  return make_pair(success, error);
}

#endif //PDB_PDBDISTRIBUTEDSTORAGETEMPLATE_H
