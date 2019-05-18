//
// Created by dimitrije on 3/8/19.
//

#include <ExecutionServerBackend.h>
#include <HeapRequestHandler.h>
#include "PDBStorageManagerBackend.h"
#include "SimpleRequestResult.h"
#include "ExRunJob.h"
#include "ExJob.h"
#include "SharedEmployee.h"

void pdb::ExecutionServerBackend::registerHandlers(pdb::PDBServer &forMe) {

  forMe.registerHandler(
      ExJob_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::ExJob>>(
          [&](Handle<pdb::ExJob> request, PDBCommunicatorPtr sendUsingMe) {

            std::string error;

            /// 1. Do the setup

            // setup an allocation block of the size of the compute plan + 1MB so we can do the setup and build the pipeline
            const UseTemporaryAllocationBlock tempBlock{request->computationSize + 2 * 1024};

            // grab the storage manager
            auto storage = this->getFunctionalityPtr<PDBStorageManagerBackend>();

            // setup the algorithm
            bool success = request->physicalAlgorithm->setup(storage, request, error);

            // create an allocation block to hold the response
            pdb::Handle<pdb::SimpleRequestResult> response = pdb::makeObject<pdb::SimpleRequestResult>(success, error);

            // sends result to requester
            sendUsingMe->sendObject(response, error);

            /// 2. Do the run

            // make an allocation block
            {
              // want this to be destroyed
              Handle<pdb::ExRunJob> result = sendUsingMe->getNextObject<pdb::ExRunJob> (success, error);
              if (!success || !(result->shouldRun)) {

                // cleanup the algorithm
                request->physicalAlgorithm->cleanup();

                // we are done here does not work
                return make_pair(true, error); // TODO different error message if result->shouldRun is false?
              }
            }

            // run the algorithm
            request->physicalAlgorithm->run(storage);

            // sends result to requester
            sendUsingMe->sendObject(response, error);

            // cleanup the algorithm
            request->physicalAlgorithm->cleanup();

            // just finish
            return make_pair(true, error);
          }));
}

void pdb::ExecutionServerBackend::init() {}
