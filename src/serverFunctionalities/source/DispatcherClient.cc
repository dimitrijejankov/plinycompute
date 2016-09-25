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
#ifndef OBJECTQUERYMODEL_DISPATCHERCLIENT_CC
#define OBJECTQUERYMODEL_DISPATCHERCLIENT_CC

#include "DispatcherClient.h"

namespace pdb {

    DispatcherClient::DispatcherClient(int portIn, std :: string addressIn, PDBLoggerPtr myLoggerIn) :
            port(portIn), address(addressIn), logger(myLoggerIn) {}
    DispatcherClient::~DispatcherClient() {}

    void DispatcherClient::registerHandlers (PDBServer &forMe) {} // no-op

    bool DispatcherClient::registerSet(std::pair<std::string, std::string> setAndDatabase) {
        // TODO: Implement this
        std::cout << "Registering partition policy for " << port << " : " << address << std::endl;
        return 1;
    }

}

#include "StorageClientTemplate.cc"

#endif

