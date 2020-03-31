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
          [&](const Handle<pdb::ExJob>& request, const PDBCommunicatorPtr& sendUsingMe) {

            bool success = true;
            std::string error;

            /// 1. Do the setup

            // setup an allocation block of the size of the compute plan + 1MB so we can do the setup and build the pipeline
            const UseTemporaryAllocationBlock tempBlock{request->computationSize + 2 * 1024};

            // create a response
            pdb::Handle<pdb::SimpleRequestResult> response = pdb::makeObject<pdb::SimpleRequestResult>(success, error);

            // grab the storage manager
            auto storage = this->getFunctionalityPtr<PDBStorageManagerBackend>();

            // this gives us the initial state
            auto state = request->physicalAlgorithm->getInitialState(request);

            // reset the algorithm stages
            request->physicalAlgorithm->resetStages();

            /// 2. Run all the stages

            // go through each stage and run it
            PDBPhysicalAlgorithmStagePtr stage;
            for(;;) {

              /// 2.0 Get the next algorithm

              stage = request->physicalAlgorithm->getNextStage(state);
              if(stage == nullptr) {
                break;
              }

              /// 2.1 Setup the stage

              // setup the stage
              success = stage->setup(request, state, storage, error);

              // if we failed update the response and send failure
              if(!success) {

                // update the response
                response->res = success;
                response->errMsg = error;
              }

              // sends result to requester
              sendUsingMe->sendObject(response, error);

              // make an allocation block to receive @see ExRunJob
              {
                // want this to be destroyed
                Handle<pdb::ExRunJob> result = sendUsingMe->getNextObject<pdb::ExRunJob> (success, error);
                if (!success || !(result->shouldRun)) {

                  // cleanup the algorithm
                  stage->cleanup(state);

                  // we are done here does not work
                  return make_pair(true, error); // TODO different error message if result->shouldRun is false?
                }
              }

              /// 2.2 Run the stage

              // run the algorithm
              stage->run(request, state, storage, error);

              // if we failed update the response and send failure
              if(!success) {

                // update the response
                response->res = success;
                response->errMsg = error;
              }

              // sends result to requester
              sendUsingMe->sendObject(response, error);

              /// 2.3 Do the cleanup

              // cleanup the stage
              stage->cleanup(state);
            }

            // just finish
            return make_pair(true, error);
          }));
}

void pdb::ExecutionServerBackend::init() {}
