/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#ifndef SIMPLE_REQUEST_CC
#define SIMPLE_REQUEST_CC

#include <functional>
#include <string>

#include "InterfaceFunctions.h"
#include "UseTemporaryAllocationBlock.h"
#include "PDBCommunicator.h"

using std::function;
using std::string;

#ifndef MAX_RETRIES
#define MAX_RETRIES 5
#endif
#define BLOCK_HEADER_SIZE 20
namespace pdb {

template <class CommunicatorType, class RequestType, class ResponseType, class ReturnType, class... RequestTypeParams>
ReturnType heapRequest(PDBLoggerPtr myLogger,
                       int port,
                       std::string address,
                       ReturnType onErr,
                       size_t bytesForRequest,
                       function<ReturnType(Handle<ResponseType>)> processResponse,
                       RequestTypeParams&&... args) {


    // try multiple times if we fail to connect
    int numRetries = 0;
    while (numRetries <= MAX_RETRIES) {

        // used for error handling
        string errMsg;
        bool success;

        // connect to the server
        CommunicatorType temp;
        if (temp.connectToInternetServer(myLogger, port, address, errMsg)) {

            // log the error
            myLogger->error(errMsg);
            myLogger->error("Can not connect to remote server with port=" + std::to_string(port) + " and address=" + address + ");");

            return onErr;
        }

        // log that we are connected
        myLogger->info(std::string("Successfully connected to remote server with port=") + std::to_string(port) + std::string(" and address=") + address);

        // check if it is invalid
        if (bytesForRequest <= BLOCK_HEADER_SIZE) {

            // ok this is an unrecoverable error
            myLogger->error("Too small buffer size for processing simple request");
            return onErr;
        }

        // make a block to send the request
        const UseTemporaryAllocationBlock tempBlock{bytesForRequest};

        // make the request
        Handle<RequestType> request = makeObject<RequestType>(args...);

        // send the object
        if (!temp.sendObject(request, errMsg)) {

            // yeah something happened
            myLogger->error(errMsg);
            myLogger->error("Not able to send request to server.\n");

            // retry
            numRetries++;
            continue;
        }

        // log that the object is sent
        myLogger->info("Object sent.");

        // get the response and process it
        ReturnType finalResult;
        size_t objectSize = temp.getSizeOfNextObject();

        // check if we did get a response
        if (objectSize == 0) {

            // ok we did not that sucks log what happened
            myLogger->error("We did not get a response.\n");

            // retry
            numRetries++;
            continue;
        }

        // allocate the memory
        std::unique_ptr<char[]> memory(new char[objectSize]);
        if (memory == nullptr) {


            errMsg = "FATAL ERROR in heapRequest: Can't allocate memory";
            myLogger->error(errMsg);
            std::cout << errMsg << std::endl;
            exit(-1);
        }

        {
            Handle<ResponseType> result =  temp.template getNextObject<ResponseType> (memory.get(), success, errMsg);
            if (!success) {
                myLogger->error(errMsg);
                myLogger->error("heapRequest: not able to get next object over the wire.\n");
                /// JiaNote: we need free memory here !!!

                if (numRetries < MAX_RETRIES) {
                    numRetries++;
                    continue;
                } else {
                    return onErr;
                }
            }

            finalResult = processResponse(result);
        }
        return finalResult;
    }

    //
    return onErr;
}

template <class RequestType, class SecondRequestType, class ResponseType, class ReturnType>
ReturnType simpleDoubleRequest(PDBLoggerPtr myLogger,
                               int port,
                               std::string address,
                               ReturnType onErr,
                               size_t bytesForRequest,
                               function<ReturnType(Handle<ResponseType>)> processResponse,
                               Handle<RequestType>& firstRequest,
                               Handle<SecondRequestType>& secondRequest) {

    PDBCommunicator temp;
    string errMsg;
    bool success;

    if (temp.connectToInternetServer(myLogger, port, address, errMsg)) {
        myLogger->error(errMsg);
        myLogger->error("heapRequest: not able to connect to server.\n");
        // return onErr;
        std::cout << "ERROR: can not connect to remote server with port=" << port
                  << " and address=" << address << std::endl;
        return onErr;
    }
    std::cout << "Successfully connected to remote server with port=" << port
              << " and address=" << address << std::endl;
    // build the request
    if (!temp.sendObject(firstRequest, errMsg)) {
        myLogger->error(errMsg);
        myLogger->error("simpleDoubleRequest: not able to send first request to server.\n");
        return onErr;
    }

    if (!temp.sendObject(secondRequest, errMsg)) {
        myLogger->error(errMsg);
        myLogger->error("simpleDoubleRequest: not able to send second request to server.\n");
        return onErr;
    }

    // get the response and process it
    ReturnType finalResult;
    void* memory = malloc(temp.getSizeOfNextObject());
    {
        Handle<ResponseType> result = temp.getNextObject<ResponseType>(memory, success, errMsg);
        if (!success) {
            myLogger->error(errMsg);
            myLogger->error("heapRequest: not able to get next object over the wire.\n");
            return onErr;
        }

        finalResult = processResponse(result);
    }
    free(memory);
    return finalResult;
}
}
#endif
