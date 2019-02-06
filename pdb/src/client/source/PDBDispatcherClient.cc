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

#include "PDBDispatcherClient.h"
#include "HeapRequest.h"
#include "DispatcherRegisterPartitionPolicy.h"

namespace pdb {

PDBDispatcherClient::PDBDispatcherClient() = default;

PDBDispatcherClient::~PDBDispatcherClient() = default;

PDBDispatcherClient::PDBDispatcherClient(int portIn, std::string addressIn, PDBLoggerPtr myLoggerIn) {
    this->logger = myLoggerIn;
    this->port = portIn;
    this->address = addressIn;
}

void PDBDispatcherClient::registerHandlers(PDBServer& forMe) {}

}

#include "StorageClientTemplate.cc"

#endif
