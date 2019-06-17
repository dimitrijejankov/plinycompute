//
// Created by dimitrije on 2/20/19.
//

#include <PDBComputationClient.h>
#include <HeapRequest.h>
#include <CSExecuteComputation.h>
#include <SimpleRequestResult.h>

pdb::PDBComputationClient::PDBComputationClient(const string &address, int port, const pdb::PDBLoggerPtr &myLogger)
    : address(address), port(port), myLogger(myLogger) {

  // get the communicator information
  this->port = port;
  this->address = address;
  this->myLogger = myLogger;
}

bool pdb::PDBComputationClient::executeComputations(Handle<Vector<Handle<Computation>>> &computations, const pdb::String &tcap, std::string &error) {

  // essentially the buffer should be of this size //TODO this needs to be stress tested
  auto bufferSize = getRecord(computations)->numBytes() + tcap.size() + 1024 * 2;

  // increment the buffer in increments of
  while(bufferSize < MAX_COMPUTATION_SIZE) {

    try {

        // send the request
        return RequestFactory::heapRequest<CSExecuteComputation, SimpleRequestResult, bool>(myLogger, port, address, false, bufferSize,
        [&](Handle<SimpleRequestResult> result) {

        // check the response
        if ((result != nullptr && !result->getRes().first) || result == nullptr) {

        // log the error
        myLogger->error("Error executing computations: " + result->getRes().second);
        error = "Error executing computations: " + result->getRes().second;

        // we are done here
        return false;
        }

        // awesome we finished
        return true;
        }, computations, tcap, bufferSize);
    }
    catch(pdb::NotEnoughSpace &n) {

      // increment the buffer
      bufferSize += MAX_STEP_SIZE;

    }
  }

  // finish since the computation was just too large
  return false;
}

// set the constants
const uint64_t pdb::PDBComputationClient::MAX_COMPUTATION_SIZE = 100u * 1024u * 1024u;
const uint64_t pdb::PDBComputationClient::MAX_STEP_SIZE = 10u * 1024u * 1024u;