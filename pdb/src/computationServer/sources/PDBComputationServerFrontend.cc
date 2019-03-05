//
// Created by dimitrije on 2/19/19.
//

#include <boost/filesystem/path.hpp>
#include <ExJob.h>
#include <PDBComputationServerFrontend.h>
#include <GenericWork.h>

#include "PDBComputationServerFrontend.h"
#include "HeapRequestHandler.h"
#include "CSExecuteComputation.h"
#include "PDBPhysicalOptimizer.h"
#include "PDBDistributedStorage.h"
#include "ExRunJob.h"

void pdb::PDBComputationServerFrontend::init() {

  // init the class
  logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "logs").string(),
                                       "PDBComputationServerFrontend.log");
}

bool pdb::PDBComputationServerFrontend::executeJob(pdb::Handle<pdb::ExJob> &job) {

  atomic_bool success;
  success = true;

  // grab the nodes we want to forward the request to
  auto nodes = getFunctionality<PDBCatalogClient>().getActiveWorkerNodes();

  // create the buzzer
  atomic_int counter;
  counter = 0;
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if(myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // for each node start a worker
  for(const auto &node : nodes) {

    /// 0. Start a worker for each node

    // grab a worker
    auto worker = parent->getWorkerQueue()->getWorker();

    // make the work
    PDBWorkPtr myWork = make_shared<pdb::GenericWork>([=, &counter, &job](PDBBuzzerPtr callerBuzzer) {

      std::string errMsg;

      /// 1. Connect to the node

      // connect to the server
      PDBCommunicator comm;
      size_t numRetries = 0;
      while (comm.connectToInternetServer(logger, node->port, node->address, errMsg)) {

        // log the error
        logger->error(errMsg);
        logger->error("Can not connect to remote server with port=" + std::to_string(node->port) + " and address=" + node->address + ");");

        // retry
        numRetries++;
        if(numRetries < getConfiguration()->maxRetries) {
          continue;
        }

        // ok we are done here since we are out of retries signal an error
        callerBuzzer->buzz(PDBAlarm::GenericError, counter);
        return;
      }

      /// 2. schedule the computation
      if(!scheduleJob(comm, job, errMsg)) {

        // we failed to schedule the job
        callerBuzzer->buzz(PDBAlarm::GenericError, counter);
        return;
      }

      /// 3. Run the computation and wait for it to finish
      if(!runScheduledJob(comm, errMsg)) {

        // we failed to run the job
        callerBuzzer->buzz(PDBAlarm::GenericError, counter);
        return;
      }

      // excellent everything worked just as expected
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // wait until all the nodes are finished
  while (counter < nodes.size()) {
    tempBuzzer->wait();
  }

  return success;
}

bool pdb::PDBComputationServerFrontend::scheduleJob(pdb::PDBCommunicator &temp, pdb::Handle<pdb::ExJob> &job, std::string &errMsg) {

  /// 1. Send the computation

  // send the object
  if (!temp.sendObject<pdb::ExJob>(job, errMsg, job->computationSize + 1024 * 1024)) {

    // yeah something happened
    logger->error(errMsg);
    logger->error("Not able to send request to server.\n");

    // we are done here we do not recover from this error
    return false;
  }

  /// 2. Wait for ACK

  // get the response and process it
  size_t objectSize = temp.getSizeOfNextObject();

  // check if we did get a response
  if (objectSize == 0) {

    // ok we did not that sucks log what happened
    logger->error("We did not get a response.\n");

    // we are done here we do not recover from this error
    return false;
  }

  // allocate the memory
  std::unique_ptr<char[]> memory(new char[objectSize]);
  if (memory == nullptr) {

    errMsg = "FATAL ERROR: Can't allocate memory";
    logger->error(errMsg);

    /// TODO this needs to be an exception or something
    // this is a fatal error we should not be running out of memory
    exit(-1);
  }

  {
    bool success;

    // want this to be destroyed
    Handle<SimpleRequestResult> result =  temp.getNextObject<SimpleRequestResult> (memory.get(), success, errMsg);
    if (!success) {

      // log the error
      logger->error(errMsg);
      logger->error("not able to get next object over the wire.\n");

      // we are done here does not work
      return false;
    }
  }

  // return true
  return true;
}

bool pdb::PDBComputationServerFrontend::runScheduledJob(pdb::PDBCommunicator &communicator, string &errMsg) {

  // make an allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{1024};

  // make a request
  pdb::Handle<ExRunJob> request = pdb::makeObject<ExRunJob>();

  /// 1. Send the request

  // send the object
  if (!communicator.sendObject<pdb::ExRunJob>(request, errMsg)) {

    // yeah something happened
    logger->error(errMsg);
    logger->error("Not able to job to server.\n");

    // we are done here we do not recover from this error
    return false;
  }

  /// 2. Wait for the job to finish

  // get the response and process it
  size_t objectSize = communicator.getSizeOfNextObject();

  // check if we did get a response
  if (objectSize == 0) {

    // ok we did not that sucks log what happened
    logger->error("We did not get a response for the job request.\n");

    // we are done here we do not recover from this error
    return false;
  }

  {
    bool success;

    // want this to be destroyed
    Handle<SimpleRequestResult> result = communicator.getNextObject<SimpleRequestResult> (success, errMsg);
    if (!success) {

      // log the error
      logger->error(errMsg);
      logger->error("Did not get the response that we are done running this.\n");

      // we are done here does not work
      return false;
    }
  }

  // return true
  return true;
}

void pdb::PDBComputationServerFrontend::registerHandlers(pdb::PDBServer &forMe) {

  forMe.registerHandler(
      CSExecuteComputation_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::CSExecuteComputation>>(
          [&](Handle<pdb::CSExecuteComputation> request, PDBCommunicatorPtr sendUsingMe) {

            /// 1. Init the optimizer

            // indicators
            bool success = true;
            std::string error;

            // the id associated with this computation
            auto compID = this->statsManager.startComputation();

            // distributed storage
            auto catalogClient = getFunctionalityPtr<pdb::PDBCatalogClient>();

            // init the optimizer
            pdb::PDBPhysicalOptimizer optimizer(request->tcapString, catalogClient, logger);

            // we start from job 0
            uint64_t jobID = 0;

            /// 2. Run job while the optimizer can spit out an algorithm

            // while we still have jobs to execute
            while(optimizer.hasAlgorithmToRun()) {

              // make an allocation block the computation size + 1MB for algorithm and stuff
              const pdb::UseTemporaryAllocationBlock tempBlock{request->numBytes + 1024*1024};

              // grab a algorithm
              auto algorithm = optimizer.getNextAlgorithm();

              // make the job
              Handle<ExJob> job = pdb::makeObject<ExJob>();

              // set the job stuff
              job->computationID = compID;
              job->computations = request->computations;
              job->tcap = request->tcapString;
              job->jobID = jobID++;
              job->physicalAlgorithm = algorithm;

              // just set how much we need for the computation object in case somebody embed some data in it
              job->computationSize = request->numBytes;

              // broadcast the job to each node and run it...
              if(!executeJob(job)) {

                // we failed therefore we are done here
                success = false;
                error = "We failed to execute the job with the ID (" + std::to_string(job->jobID) + + ")";

                break;
              }

              // update stats
              optimizer.updateStats();
            }

            // end the computation
            this->statsManager.endComputation(compID);

            /// 3. Send the result of the execution back to the client

            // make an allocation block
            const pdb::UseTemporaryAllocationBlock tempBlock{1024};

            // create an allocation block to hold the response
            pdb::Handle<pdb::SimpleRequestResult> response = pdb::makeObject<pdb::SimpleRequestResult>(success, error);

            // sends result to requester
            sendUsingMe->sendObject(response, error);

            return make_pair(success, error);
          }));

}

