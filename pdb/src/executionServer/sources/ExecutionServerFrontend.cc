//
// Created by dimitrije on 3/4/19.
//

#include <ExJob.h>
#include <HeapRequestHandler.h>
#include <ExRunJob.h>
#include <PDBStorageManagerBackend.h>
#include <SharedEmployee.h>
#include <boost/filesystem/path.hpp>
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

            /// 1. Connect to the backend and forward the request

            PDBCommunicatorPtr communicatorToBackend = make_shared<PDBCommunicator>();
            if (!communicatorToBackend->connectToLocalServer(logger, getConfiguration()->ipcFile, error)) {

              // log the error
              logger->error(error);

              // create an allocation block to hold the response
              pdb::Handle<pdb::SimpleRequestResult> response = pdb::makeObject<pdb::SimpleRequestResult>(false, error);

              // sends result to requester
              sendUsingMe->sendObject(response, error);

              // return error
              return std::make_pair(false, error);
            }

            /// 2. Forward the request

            // send the object with a buffer of 1mb + computation size should be enough
            if(!communicatorToBackend->sendObject(request, error, request->computationSize + 1024 * 1024)) {

              // we failed to send a response
              pdb::Handle<pdb::SimpleRequestResult> response = pdb::makeObject<pdb::SimpleRequestResult>(false, error);

              // sends result to requester
              sendUsingMe->sendObject(response, error);

              // return error
              return std::make_pair(false, error);
            }

            /// 3. wait for all the stages to finish
            for(int i = 0; i < request->physicalAlgorithm->numStages(); ++i) {

              /// 3.1 Wait for ACK that the algorithm is setup
              success = RequestFactory::waitHeapRequest<SimpleRequestResult, bool>(logger, communicatorToBackend, false,
[&](const Handle<SimpleRequestResult>& result) {

                // check the result
                if (result != nullptr && result->getRes().first) {
                  return true;
                }

                // log the error
                error = "Error response from distributed-storage: " + result->getRes().second;
                logger->error("Error response from distributed-storage: " + result->getRes().second);

                return false;
              });

              /// 3.2 Forward the ack to the computation server

              // create an allocation block to hold the response
              pdb::Handle<pdb::SimpleRequestResult> response = pdb::makeObject<pdb::SimpleRequestResult>(success, error);

              // sends result to requester
              sendUsingMe->sendObject(response, error);

              // did we fail if so finish this
              if(!success) {

                // make the request to abort to backend server
                pdb::Handle<ExRunJob> abortRequest = pdb::makeObject<ExRunJob>(false);

                // send the abort request
                success = communicatorToBackend->sendObject(abortRequest, error, 1024);

                // create the response for the computation server
                response = pdb::makeObject<pdb::SimpleRequestResult>(false, error);

                // sends result to computation server
                sendUsingMe->sendObject(response, error);

                // we failed so we are done here
                break;
              }

              /// 3.3 Wait now for the request from the computation server to run the computation

              // want this to be destroyed
              Handle<pdb::ExRunJob> result = sendUsingMe->getNextObject<pdb::ExRunJob> (success, error);
              if (!success) {

                // we failed so get out of here
                break;
              }

              /// 3.4 Forward the request to the backend, so we can start the to run the algorithm

              success = communicatorToBackend->sendObject(result, error, 1024);

              // if we failed send a request back that we did
              if(!success) {

                // we failed therefore we are out of here
                break;
              }

              /// 3.5 Wait for the backend to respond

              success = RequestFactory::waitHeapRequest<SimpleRequestResult, bool>(logger, communicatorToBackend, false,
                  [&](const Handle<SimpleRequestResult>& result) {

               // check the result
               if (result != nullptr && result->getRes().first) {
                 return true;
               }

               // log the error
               error = "Error response from distributed-storage: " + result->getRes().second;
               logger->error("Error response from distributed-storage: " + result->getRes().second);

               return false;
              });

              // if we failed break out of the loop so that we can respond to the computation server
              if(!success) {

                // we failed therefore we are out of here
                break;
              }

              /// 3.2 Forward status the computation server

              // create an allocation block to hold the response
              response->res = success;
              response->errMsg = error;

              // sends result to requester
              sendUsingMe->sendObject(response, error);

              // did we fail if so finish this
              if(!success) {

                // make the request to abort to backend server
                pdb::Handle<ExRunJob> abortRequest = pdb::makeObject<ExRunJob>(false);

                // send the abort request
                success = communicatorToBackend->sendObject(abortRequest, error, 1024);

                // create the response for the computation server
                response = pdb::makeObject<pdb::SimpleRequestResult>(false, error);

                // sends result to computation server
                sendUsingMe->sendObject(response, error);

                // we failed so we are done here
                break;
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
