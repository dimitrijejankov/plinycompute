#include <utility>


#include <StoStartFeedingPageSetRequest.h>

//
// Created by dimitrije on 4/5/19.
//

#include <PDBPageNetworkSender.h>
#include <StoStartFeedingPageSetRequest.h>
#include <UseTemporaryAllocationBlock.h>

#include "PDBPageNetworkSender.h"

pdb::PDBPageNetworkSender::PDBPageNetworkSender(string address, int32_t port, uint64_t maxRetries, PDBLoggerPtr logger,
                                                std::pair<uint64_t, std::string> pageSetID, pdb::PDBPageQueuePtr queue)
    : address(std::move(address)), port(port), queue(std::move(queue)), logger(std::move(logger)), pageSetID(std::move(pageSetID)), maxRetries(maxRetries) {}

bool pdb::PDBPageNetworkSender::setup() {

  std::string errMsg;

  // connect to the server
  size_t numRetries = 0;
  comm = std::make_shared<PDBCommunicator>();
  while (!comm->connectToInternetServer(logger, port, address, errMsg)) {

    // log the error
    logger->error(errMsg);
    logger->error("Can not connect to remote server with port=" + std::to_string(port) + " and address=" + address + ");");

    // retry
    numRetries++;
    if(numRetries < maxRetries) {
      continue;
    }

    // finish here since we are out of retries
    return false;
  }

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // make the request
  Handle<StoStartFeedingPageSetRequest> request = makeObject<StoStartFeedingPageSetRequest>(pageSetID);

  // send the object
  if (!comm->sendObject(request, errMsg)) {

    // yeah something happened
    logger->error(errMsg);
    logger->error("Not able to send request to server.\n");

    // we are done here we do not recover from this error
    return false;
  }

  return true;
}

bool pdb::PDBPageNetworkSender::run() {

  return true;
}
