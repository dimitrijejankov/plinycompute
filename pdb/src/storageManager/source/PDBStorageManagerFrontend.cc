//
// Created by dimitrije on 2/9/19.
//

#include <PDBStorageManagerFrontend.h>
#include <HeapRequestHandler.h>
#include <StoDispatchData.h>
#include <PDBBufferManagerInterface.h>
#include <PDBBufferManagerFrontEnd.h>
#include <StoStoreOnPageRequest.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <fstream>
#include <HeapRequest.h>
#include <StoGetNextPageRequest.h>
#include <StoGetNextPageResult.h>
#include "CatalogServer.h"
#include <StoGetPageRequest.h>
#include <StoGetPageResult.h>

namespace fs = boost::filesystem;

pdb::PDBStorageManagerFrontend::~PDBStorageManagerFrontend() {

  // open the output file
  std::ofstream ofs;
  ofs.open((boost::filesystem::path(getConfiguration()->rootDirectory) / "storage.pdb").string(),
           ios::binary | std::ofstream::out | std::ofstream::trunc);

  unsigned long numSets = lastPages.size();
  ofs.write((char *) &numSets, sizeof(unsigned long));

  // serialize the stuff
  for (auto &it : lastPages) {

    // write the database name
    unsigned long size = it.first->getDBName().size();
    ofs.write((char *) &size, sizeof(unsigned long));
    ofs.write(it.first->getDBName().c_str(), size);

    // write the set name
    size = it.first->getSetName().size();
    ofs.write((char *) &size, sizeof(unsigned long));
    ofs.write(it.first->getSetName().c_str(), size);

    // write the page stats
    ofs.write(reinterpret_cast<char *>(&it.second), sizeof(it.second));
  }

  ofs.close();
}

void pdb::PDBStorageManagerFrontend::init() {

  // init the class
  logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "logs").string(),
                                       "PDBStorageManagerFrontend.log");

  // do we have the metadata for the storage
  if (fs::exists(boost::filesystem::path(getConfiguration()->rootDirectory) / "storage.pdb")) {

    // open if stream
    std::ifstream ifs;
    ifs.open((boost::filesystem::path(getConfiguration()->rootDirectory) / "storage.pdb").string(),
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

      // read the number of pages
      PDBStorageSetStats pageStats{};
      ifs.read(reinterpret_cast<char *>(&pageStats), sizeof(pageStats));

      // store the set info
      auto set = std::make_shared<PDBSet>(dbName, setName);
      lastPages[set] = pageStats;
    }

    // close
    ifs.close();
  }
}

void pdb::PDBStorageManagerFrontend::registerHandlers(PDBServer &forMe) {

  forMe.registerHandler(
      StoGetPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoGetPageRequest>>([&](Handle<pdb::StoGetPageRequest> request,
                                                                       PDBCommunicatorPtr sendUsingMe) {
        // handle the get page request
        return handleGetPageRequest(request, sendUsingMe);
      }));

  forMe.registerHandler(
      StoDispatchData_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoDispatchData>>(
          [&](Handle<pdb::StoDispatchData> request, PDBCommunicatorPtr sendUsingMe) {

            return handleDispatchedData<PDBCommunicator, RequestFactory>(request, sendUsingMe);
          }));

  forMe.registerHandler(
      StoSetStatsRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoSetStatsRequest>>([&](pdb::Handle<pdb::StoSetStatsRequest> request, PDBCommunicatorPtr sendUsingMe) {
        return handleGetSetStats<PDBCommunicator, RequestFactory>(request, sendUsingMe);
      }));
}


