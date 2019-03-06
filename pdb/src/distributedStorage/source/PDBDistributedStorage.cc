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

#ifndef DISPATCHER_SERVER_CC
#define DISPATCHER_SERVER_CC

#include "PDBDistributedStorage.h"
#include <snappy.h>
#include <HeapRequestHandler.h>
#include <DisAddData.h>
#include <BufGetPageRequest.h>
#include <PDBBufferManagerInterface.h>
#include <PDBDispatchRandomPolicy.h>
#include <StoDispatchData.h>
#include "PDBCatalogClient.h"
#include <boost/filesystem/path.hpp>
#include <PDBDistributedStorage.h>
#include <fstream>
#include <boost/filesystem/operations.hpp>

namespace pdb {

namespace fs = boost::filesystem;


void PDBDistributedStorage::init() {

  // init the policy
  policy = std::make_shared<PDBDispatchRandomPolicy>();

  // init the class
  logger = make_shared<pdb::PDBLogger>((fs::path(getConfiguration()->rootDirectory) / "logs").string(), "PDBDistributedStorage.log");
}

void PDBDistributedStorage::registerHandlers(PDBServer &forMe) {

forMe.registerHandler(
    StoGetNextPageRequest_TYPEID,
    make_shared<pdb::HeapRequestHandler<pdb::StoGetNextPageRequest>>(
        [&](Handle<pdb::StoGetNextPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

          return handleGetNextPage<PDBCommunicator, RequestFactory>(request, sendUsingMe);
        }));

forMe.registerHandler(
    DisAddData_TYPEID,
    make_shared<HeapRequestHandler<pdb::DisAddData>>(
        [&](Handle<pdb::DisAddData> request, PDBCommunicatorPtr sendUsingMe) {

          return handleAddData<PDBCommunicator, RequestFactory>(request, sendUsingMe);
    }));
}

}

#endif
