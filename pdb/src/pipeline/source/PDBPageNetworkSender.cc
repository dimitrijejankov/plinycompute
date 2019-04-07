#include <utility>


#include <StoStartFeedingPageSetRequest.h>

//
// Created by dimitrije on 4/5/19.
//

#include <PDBPageNetworkSender.h>
#include <StoStartFeedingPageSetRequest.h>
#include <UseTemporaryAllocationBlock.h>
#include <SimpleRequestResult.h>
#include <StoFeedPageRequest.h>

#include "PDBPageNetworkSender.h"

pdb::PDBPageNetworkSender::PDBPageNetworkSender(string address, int32_t port, uint64_t numberOfProcessingThreads, uint64_t numberOfNodes,
                                                uint64_t maxRetries, PDBLoggerPtr logger, std::pair<uint64_t, std::string> pageSetID, pdb::PDBPageQueuePtr queue)
    : address(std::move(address)), port(port), queue(std::move(queue)), numberOfProcessingThreads(numberOfProcessingThreads),
      numberOfNodes(numberOfNodes), logger(std::move(logger)), pageSetID(std::move(pageSetID)), maxRetries(maxRetries) {}

bool pdb::PDBPageNetworkSender::setup() {

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

  {
    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};

    // make the request
    Handle<StoStartFeedingPageSetRequest> request = makeObject<StoStartFeedingPageSetRequest>(pageSetID, numberOfProcessingThreads, numberOfNodes);

    // send the object
    if (!comm->sendObject(request, errMsg)) {

      // yeah something happened
      logger->error(errMsg);
      logger->error("Not able to send request to server.\n");

      // we are done here we do not recover from this error
      return false;
    }
  }

  // want this to be destroyed
  bool success;
  Handle<pdb::SimpleRequestResult> result = comm->getNextObject<pdb::SimpleRequestResult> (success, errMsg);
  if (success && result != nullptr) {

    // we are done here
    return result->getRes().first;
  }

  return false;
}

bool pdb::PDBPageNetworkSender::run() {

  // create an allocation block to hold the response
  const UseTemporaryAllocationBlock tempBlock{1024};

  // make the request
  Handle<pdb::StoFeedPageRequest> request = makeObject<pdb::StoFeedPageRequest>();

  // send the pages
  PDBPageHandle page;
  do {

    // get a page
    queue->wait_dequeue(page);

    // if we got a page from the queue
    if(page != nullptr) {

      // signal that we have another page
      request->hasNextPage = true;
      request->pageSize = page->getSize();

      // send the object
      if (!comm->sendObject(request, errMsg)) {
        return false;
      }

      // repin the page
      page->repin();

      // ret the record
      auto curRec = (Record<Object> *) page->getBytes();

      // get how large it was
      auto numBytes = curRec->numBytes();

      // send the page
      comm->sendBytes(page->getBytes(), numBytes, errMsg);
    }

  } while (page != nullptr);

  // signal that we are done
  request->hasNextPage = false;
  return comm->sendObject(request, errMsg);
}
