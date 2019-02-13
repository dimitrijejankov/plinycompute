//
// Created by dimitrije on 2/9/19.
//

#include <StorageManagerFrontend.h>
#include <HeapRequestHandler.h>
#include <DispDispatchData.h>
#include <PDBBufferManagerInterface.h>
#include <PDBBufferManagerFrontEnd.h>
#include <StoStoreOnPageRequest.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <fstream>

namespace fs = boost::filesystem;

pdb::StorageManagerFrontend::~StorageManagerFrontend() {

  // open the output file
  std::ofstream ofs;
  ofs.open ((boost::filesystem::path(getConfiguration()->rootDirectory) / "storage.pdb").string(), ios::binary | std::ofstream::out | std::ofstream::trunc);

  unsigned long numSets = lastPages.size();
  ofs.write((char*) &numSets, sizeof(unsigned long));

  // serialize the stuff
  for(auto &it : lastPages) {

    // write the database name
    unsigned long size = it.first->getDBName().size();
    ofs.write((char*) &size, sizeof(unsigned long));
    ofs.write(it.first->getDBName().c_str(), size);

    // write the set name
    size = it.first->getSetName().size();
    ofs.write((char*) &size, sizeof(unsigned long));
    ofs.write(it.first->getSetName().c_str(), size);

    // write the number of pages
    ofs.write((char*) &it.second, sizeof(unsigned long));
  }

  ofs.close();
}

void pdb::StorageManagerFrontend::init() {

  // init the class
  logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "logs").string(), "PDBStorageManagerFrontend.log");

  // do we have the metadata for the storage
  if (fs::exists(boost::filesystem::path(getConfiguration()->rootDirectory) / "storage.pdb")) {

    // open if stream
    std::ifstream ifs;
    ifs.open((boost::filesystem::path(getConfiguration()->rootDirectory) / "storage.pdb").string(),
             ios::binary | std::ifstream::in);

    size_t numSets;
    ifs.read((char *) &numSets, sizeof(unsigned long));

    for(int i = 0; i < numSets; ++i) {

      // read the database name
      unsigned long size;
      ifs.read((char *) &size, sizeof(unsigned long));
      std::unique_ptr<char[]> setBuffer(new char[size]);
      ifs.read(setBuffer.get(), size);
      std::string dbName(setBuffer.get(), size);

      // read the set name
      ifs.read((char *) &size, sizeof(unsigned long));
      std::unique_ptr<char[]> dbBuffer(new char[size + 1]);
      ifs.read(dbBuffer.get(), size + 1);
      std::string setName(dbBuffer.get(), size);

      // read the number of pages
      size_t pageNum;
      ifs.read((char *) &pageNum, sizeof(unsigned long));

      // store the set info
      auto set = std::make_shared<PDBSet>(setName, dbName);
      lastPages[set] = pageNum;
    }

    // close
    ifs.close();
  }
}

void pdb::StorageManagerFrontend::registerHandlers(PDBServer &forMe) {

  forMe.registerHandler(
    DispDispatchData_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::DispDispatchData>>(
      [&](Handle<pdb::DispDispatchData> request, PDBCommunicatorPtr sendUsingMe) {

        /// 1. Get the page from the dispatcher

        // the error
        std::string error;

        // grab the buffer manager
        auto bufferManager = std::dynamic_pointer_cast<PDBBufferManagerFrontEnd>(getFunctionalityPtr<PDBBufferManagerInterface>());

        // figure out how large the compressed payload is
        size_t numBytes = sendUsingMe->getSizeOfNextObject();

        // grab a page to write this
        auto page = bufferManager->getPage(numBytes);

        // grab the bytes
        auto success = sendUsingMe->receiveBytes(page->getBytes(), error);

        // did we fail
        if(!success) {

          // create an allocation block to hold the response
          const UseTemporaryAllocationBlock tempBlock{1024};
          Handle<SimpleRequestResult> response = makeObject<SimpleRequestResult>(false, error);

          // sends result to requester
          sendUsingMe->sendObject(response, error);

          return std::make_pair(false, error);
        }

        /// 2. Figure out the page we want to put this thing onto

        uint64_t pageNum;
        {
          // lock the stuff that keeps track of the last page
          unique_lock<std::mutex> lck;

          // make the set
          auto set = std::make_shared<PDBSet>(request->setName, request->databaseName);

          // try to find the set
          auto it = lastPages.find(set);

          // do we even have a record for this set
          if(it == lastPages.end()) {

            // set the page to zero since this is the first page
            lastPages[set] = 0;
            pageNum = 0;
          }
          else {

            // increment the last page
            pageNum = ++it->second;
          }
        }

        /// 3. Initiate the storing on the backend

        PDBCommunicatorPtr communicatorToBackend = make_shared<PDBCommunicator>();
        if (communicatorToBackend->connectToLocalServer(logger, getConfiguration()->ipcFile, error)) {

          return std::make_pair(false, error);
        }

        // create an allocation block to hold the response
        const UseTemporaryAllocationBlock tempBlock{1024};
        Handle<StoStoreOnPageRequest> response = makeObject<StoStoreOnPageRequest>(request->databaseName, request->setName, pageNum, request->compressedSize);

        // send the thing to the backend
        if (!communicatorToBackend->sendObject(response, error)) {

          // finish
          return std::make_pair(false, std::string("Could not send the thing to the backend"));
        }

        /// 4. Forward the page to the backend

        // forward the page
        success = bufferManager->forwardPage(page, communicatorToBackend, error);


        /// 5. Send the response that we are done

        // create an allocation block to hold the response
        Handle<SimpleRequestResult> simpleResponse = makeObject<SimpleRequestResult>(true, error);

        // sends result to requester
        sendUsingMe->sendObject(simpleResponse, error);

        // finish
        return std::make_pair(success, error);
      }));

}
