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
#include <DisDispatchData.h>
#include "PDBCatalogClient.h"
#include <boost/filesystem/path.hpp>
#include <PDBDistributedStorage.h>
#include <fstream>
#include <boost/filesystem/operations.hpp>

namespace pdb {

namespace fs = boost::filesystem;

PDBDistributedStorage::~PDBDistributedStorage() {

  // open the output file
  std::ofstream ofs;
  ofs.open((boost::filesystem::path(getConfiguration()->rootDirectory) / "distributedStorage.pdb").string(),
           ios::binary | std::ofstream::out | std::ofstream::trunc);

  unsigned long numSets = setSizes.size();
  ofs.write((char *) &numSets, sizeof(unsigned long));

  // serialize the stuff
  for (auto &it : setSizes) {

    // write the database name
    unsigned long size = it.first.first.size();
    ofs.write((char *) &size, sizeof(unsigned long));
    ofs.write(it.first.first.c_str(), size);

    // write the set name
    size = it.first.second.size();
    ofs.write((char *) &size, sizeof(unsigned long));
    ofs.write(it.first.second.c_str(), size);

    // write the set size
    ofs.write(reinterpret_cast<char *>(&it.second), sizeof(it.second));
  }

  ofs.close();
}

void PDBDistributedStorage::init() {

  // init the policy
  policy = std::make_shared<PDBDispatchRandomPolicy>();

  // init the class
  logger = make_shared<pdb::PDBLogger>((fs::path(getConfiguration()->rootDirectory) / "logs").string(), "PDBDistributedStorage.log");

  // do we have the metadata for the storage
  if (fs::exists(fs::path(getConfiguration()->rootDirectory) / "distributedStorage.pdb")) {

    // open if stream
    std::ifstream ifs;
    ifs.open((fs::path(getConfiguration()->rootDirectory) / "distributedStorage.pdb").string(),
             ios::binary | std::ifstream::in);

    size_t numSets;
    ifs.read((char *) &numSets, sizeof(unsigned long));

    for (int i = 0; i < numSets; ++i) {

      // read the database name
      unsigned long size;
      ifs.read((char *) &size, sizeof(unsigned long));
      std::unique_ptr<char[]> setBuffer(new char[size]);
      ifs.read(setBuffer.get(), size);
      std::string dbName(setBuffer.get(), size);

      // read the set name
      ifs.read((char *) &size, sizeof(unsigned long));
      std::unique_ptr<char[]> dbBuffer(new char[size]);
      ifs.read(dbBuffer.get(), size);
      std::string setName(dbBuffer.get(), size);

      // read the set size
      unsigned long setSize;
      ifs.read(reinterpret_cast<char *>(&setSize), sizeof(setSize));

      // store the set info
      setSizes[std::make_pair(dbName, setName)] = setSize;
    }

    // close
    ifs.close();
  }
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

size_t PDBDistributedStorage::getSetSize(const std::pair<std::string, std::string> &set) {

  // return the set size if it exists
  auto it = setSizes.find(set);
  if(it != setSizes.end()) {
    return it->second;
  }

  return 0;
}

}

#endif
