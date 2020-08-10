//
// Created by dimitrije on 3/4/19.
//

#include <ExJob.h>
#include <HeapRequestHandler.h>
#include <ExRunJob.h>
#include <SharedEmployee.h>
#include <boost/filesystem/path.hpp>
#include <PDBStorageManagerFrontend.h>
#include "ExecutionServerFrontend.h"
#include "SimpleRequestResult.h"

void pdb::ExecutionServerFrontend::registerHandlers(pdb::PDBServer &forMe) {

  forMe.registerHandler(
      ExJob_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::ExJob>>(
          [&](Handle<pdb::ExJob> request, const PDBCommunicatorPtr& sendUsingMe) {

            // this is where we put the error
            std::string error;
            bool success = true;

            // we will use 2 kb just to make sure there is enough space for the requests
            const UseTemporaryAllocationBlock tempBlock{2 * 1024};

            // this gives us the initial state
            auto state = request->physicalAlgorithm->getInitialState(request);

            // grab the storage manager
            auto storage = this->getFunctionalityPtr<PDBStorageManagerFrontend>();

            // go through each stage and run it
            PDBPhysicalAlgorithmStagePtr stage;
            for(int i = 0; i < request->physicalAlgorithm->numStages(); ++i) {

              /// 2.0 Get the next algorithm

              stage = request->physicalAlgorithm->getNextStage(state);
              if(stage == nullptr) {
                break;
              }

              /// 2.1 Setup the stage

              // setup the stage
              success = stage->setup(request, state, storage, error);

              /// 2.2 Send the response

              // create an allocation block to hold the response
              pdb::Handle<pdb::SimpleRequestResult> response = pdb::makeObject<pdb::SimpleRequestResult>(success, error);

              // log the error
              logger->error(error);

              // sends result to requester
              sendUsingMe->sendObject(response, error);

              if(!success) {
                // return error
                return std::make_pair(false, error);
              }

              /// 2.3 Wait now for the request from the computation server to run the computation

              // want this to be destroyed
              Handle<pdb::ExRunJob> result = sendUsingMe->getNextObject<pdb::ExRunJob> (success, error);
              if (!success || !result->shouldRun) {

                // we failed so get out of here
                break;
              }

              /// 2.4 if everything went well we simply run it

              // run it
              if(result->shouldRun) {
                success = stage->run(request, state, storage, error);
              }

            }

            // we are done here does not work
            return make_pair(success, error);
      }));
}

void pdb::ExecutionServerFrontend::init() {

  // init the class
  logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "logs").string(),
                                       "ExecutionServerFrontend.log");

}
