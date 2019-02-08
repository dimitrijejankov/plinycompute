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

#ifndef PAGED_REQUEST_HANDLER_H
#define PAGED_REQUEST_HANDLER_H

#include "PDBCommunicator.h"
#include "PDBCommWork.h"
#include "PDBServer.h"
#include "UseTemporaryAllocationBlock.h"
#include "PDBBuzzer.h"
#include <memory>
#include <PDBBufferManagerInterface.h>

/**
 * This template is used to make a simple piece of work that accepts an object of type RequestType
 * from the client,
 * processes the request, then sends the response back via a communicator.  The constructor for the class
 * takes as an argument the lambda that is to be used to process the RequestType object
 */
namespace pdb {

template <class RequestType>
class PagedRequestHandler : public PDBCommWork {

public:
    // this accepts the lambda that is used to process the RequestType object
    explicit PagedRequestHandler(std::function<std::pair<bool, std::string>(Handle<RequestType>, PDBCommunicatorPtr)> useMe) {
        processRequest = useMe;
    }

    PDBCommWorkPtr clone() override {
        return std::make_shared<PagedRequestHandler<RequestType>>(processRequest);
    }

    void execute(PDBBuzzerPtr callerBuzzer) override {

        // first, get the request
        PDBCommunicatorPtr myCommunicator = getCommunicator();
        PDBLoggerPtr myLogger = getLogger();
        bool success;
        std::string errMsg;
        size_t objectSize = myCommunicator->getSizeOfNextObject();
        myLogger->debug(std::string("SimpleRequestHandle: to receive object with size=") + std::to_string(objectSize));
        if (objectSize == 0) {
            std::cout << "HeapRequestHandler: object size=0" << std::endl;
            myLogger->error("HeapRequestHandler: object size=0");
            callerBuzzer->buzz(PDBAlarm::GenericError);
            return;
        }

        auto page = server->getFunctionality<PDBBufferManagerInterface>().getPage(myCommunicator->getSizeOfNextObject());
        void* memory = page->getBytes();
        {
            Handle<RequestType> request = myCommunicator->getNextObject<RequestType>(memory, success, errMsg);

            if (!success) {
                myLogger->error("HeapRequestHandler: tried to get the next object and failed; " + errMsg);
                callerBuzzer->buzz(PDBAlarm::GenericError);
                return;
            }

            std::pair<bool, std::string> res = processRequest(request, myCommunicator);
            if (!res.first) {
                myLogger->error("HeapRequestHandler: tried to process the request and failed; " + errMsg);
                callerBuzzer->buzz(PDBAlarm::GenericError);
                return;
            }

            myLogger->info("HeapRequestHandler: finished processing request.");
            callerBuzzer->buzz(PDBAlarm::WorkAllDone);
        }
    }

private:
    function<pair<bool, std::string>(Handle<RequestType>, PDBCommunicatorPtr)> processRequest;
};
}

#endif
