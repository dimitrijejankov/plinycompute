//
// Created by dimitrije on 2/9/19.
//

#include <PDBStorageManagerFrontend.h>
#include <HeapRequestHandler.h>
#include <DispDispatchData.h>
#include <PDBBufferManagerInterface.h>
#include <PDBBufferManagerFrontEnd.h>
#include <StoStoreOnPageRequest.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <fstream>
#include <HeapRequest.h>
#include <StoGetNextPageRequest.h>
#include <StoGetNextPageResult.h>
#include <CatalogServer.h>
#include <StoGetPageRequest.h>
#include <StoGetPageResult.h>

namespace fs = boost::filesystem;

pdb::PDBStorageManagerFrontend::~PDBStorageManagerFrontend() {

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
    ofs.write(reinterpret_cast<char*>(&it.second),sizeof(it.second));
  }

  ofs.close();
}

void pdb::PDBStorageManagerFrontend::init() {

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
      std::unique_ptr<char[]> dbBuffer(new char[size]);
      ifs.read(dbBuffer.get(), size);
      std::string setName(dbBuffer.get(), size);

      // read the number of pages
      unsigned long pageNum;
      ifs.read(reinterpret_cast<char *>(&pageNum), sizeof(pageNum));

      // store the set info
      auto set = std::make_shared<PDBSet>(setName, dbName);
      lastPages[set] = pageNum;
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

        /// 1. Check if we have the page

        // create the set identifier
        auto set = std::make_shared<PDBSet>(request->setName, request->databaseName);

        bool hasPage = false;
        // check if the set exists
        {
          // lock the stuff that keeps track of the last page
          unique_lock<std::mutex> lck;

          // check for the page
          auto it = lastPages.find(set);
          if(it != lastPages.end() && it->second >= request->page) {
            hasPage = true;
          }
        }

        /// 2. If we don't have it send an error

        if(!hasPage) {

          // make an allocation block
          const UseTemporaryAllocationBlock tempBlock{1024};

          // create an allocation block to hold the response
          Handle<StoGetPageResult> response = makeObject<StoGetPageResult>(0, false);

          // sends result to requester
          string error;
          sendUsingMe->sendObject(response, error);

          // This is an issue we simply return false only a manager can serve pages
          return std::make_pair(false, error);
        }

        /// 3. We grab the page and compress it.

        // grab the page
        auto page = getFunctionalityPtr<PDBBufferManagerInterface>()->getPage(set, request->page);

        // grab the vector
        auto* pageRecord = (Record<Vector<Handle<Object>>> *) (page->getBytes());

        // grab an anonymous page to store the compressed stuff
        auto maxCompressedSize = snappy::MaxCompressedLength(pageRecord->numBytes());
        auto compressedPage = getFunctionalityPtr<PDBBufferManagerInterface>()->getPage(maxCompressedSize);

        // compress the record
        size_t compressedSize;
        snappy::RawCompress((char*) pageRecord, pageRecord->numBytes(), (char*)compressedPage->getBytes(), &compressedSize);

        /// 4. Send the compressed page

        // make an allocation block
        const UseTemporaryAllocationBlock tempBlock{1024};

        // create an allocation block to hold the response
        Handle<StoGetPageResult> response = makeObject<StoGetPageResult>(compressedSize, true);

        // sends result to requester
        string error;
        sendUsingMe->sendObject(response, error);

        // now, send the bytes
        if (!sendUsingMe->sendBytes(compressedPage->getBytes(), compressedSize, error)) {

          logger->error(error);
          logger->error("sending page bytes: not able to send data to client.\n");

          // we are done here
          return std::make_pair(false, std::string(error));
        }

        // we are done!
        return std::make_pair(true, std::string(""));
      }));

  forMe.registerHandler(
      StoGetNextPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoGetNextPageRequest>>(
          [&](Handle<pdb::StoGetNextPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

            /// 1. Check if we are the manager

            if(!getConfiguration()->isManager) {

              std::string error = "Only a manager can serve pages!";

              // make an allocation block
              const UseTemporaryAllocationBlock tempBlock{1024};

              // create an allocation block to hold the response
              Handle<StoGetNextPageResult> response = makeObject<StoGetNextPageResult>(0, "", 0, false);

              // sends result to requester
              sendUsingMe->sendObject(response, error);

              // this is an issue we simply return false only a manager can serve pages
              return std::make_pair(false, error);
            }

            /// 2. Figure out if the node we last accessed is still there

            // sort the nodes
            auto nodes = getFunctionality<PDBCatalogClient>().getActiveWorkerNodes();
            sort(nodes.begin(), nodes.end(), [](const PDBCatalogNodePtr & a, const PDBCatalogNodePtr & b) {
              return a->nodeID > b->nodeID;
            });

            // if this is the first time we are requesting a node the next node is
            // the first node otherwise we try to find the previous node
            auto node = nodes.begin();

            if(!request->isFirst) {

              // try to find the last used node
              node = std::find_if(nodes.begin(), nodes.end(), [&] (PDBCatalogNodePtr s) { return request->nodeID == s->nodeID; } );

              // if the node is not active anymore
              if(node == nodes.end()) {

                std::string error = "Could not find the specified previous node";

                // make an allocation block
                const UseTemporaryAllocationBlock tempBlock{1024};

                // create an allocation block to hold the response
                Handle<StoGetNextPageResult> response = makeObject<StoGetNextPageResult>(0, "", 0, false);

                // sends result to requester
                sendUsingMe->sendObject(response, error);

                // This is an issue we simply return false only a manager can serve pages
                return std::make_pair(false, error);
              }
            }

            /// 3. Find the next page

            uint64_t currPage = request->page;
            for(auto it = node; it != nodes.end(); ++it) {

              // grab a page
              auto pageInfo = requestPage(*it, request->databaseName, request->setName, currPage);

              // does it have the page
              if(pageInfo.first != nullptr) {

                // make an allocation block
                const UseTemporaryAllocationBlock tempBlock{1024};

                // create an allocation block to hold the response
                Handle<StoGetNextPageResult> response = makeObject<StoGetNextPageResult>(currPage, (*it)->nodeID, pageInfo.second, true);

                // sends result to requester
                std::string error;
                sendUsingMe->sendObject(response, error);

                // now, send the bytes
                if (!sendUsingMe->sendBytes(pageInfo.first->getBytes(), pageInfo.second, error)) {

                  logger->error(error);
                  logger->error("sending page bytes: not able to send data to client.\n");

                  // we are done here
                  return std::make_pair(false, std::string(error));
                }

                // we succeeded
                return std::make_pair(true, std::string(""));
              }

              // set the page to 0 since we are grabbing a page from a new node
              currPage = 0;
            }

            // make an allocation block
            const UseTemporaryAllocationBlock tempBlock{1024};

            // create an allocation block to hold the response
            Handle<StoGetNextPageResult> response = makeObject<StoGetNextPageResult>(0, "", 0, false);

            // sends result to requester
            std::string error;
            bool success = sendUsingMe->sendObject(response, error);

            return std::make_pair(success, error);
          }));

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

        /// 5. Wait for the backend to finish the stuff

        RequestFactory::waitHeapRequest<SimpleRequestResult, bool>(logger, communicatorToBackend, false,
            [&](Handle<SimpleRequestResult> result) {

          // check the result
          if (result != nullptr && result->getRes().first) {
            return true;
          }

          // log the error
          error = "Error response from dispatcher: " + result->getRes().second;
          logger->error("Error response from dispatcher: " + result->getRes().second);

          return false;
        });

        /// 6. Send the response that we are done

        // create an allocation block to hold the response
        Handle<SimpleRequestResult> simpleResponse = makeObject<SimpleRequestResult>(true, error);

        // sends result to requester
        sendUsingMe->sendObject(simpleResponse, error);

        // finish
        return std::make_pair(success, error);
      }));

}

std::pair<pdb::PDBPageHandle, size_t> pdb::PDBStorageManagerFrontend::requestPage(const PDBCatalogNodePtr& node, const std::string &databaseName, const std::string &setName, uint64_t page) {

  // the communicator
  PDBCommunicatorPtr comm = make_shared<PDBCommunicator>();
  string errMsg;

  // try multiple times if we fail to connect
  int numRetries = 0;
  while (numRetries <= getConfiguration()->maxRetries) {

    // connect to the server
    if (comm->connectToInternetServer(logger, node->port, node->address, errMsg)) {

      // log the error
      logger->error(errMsg);
      logger->error("Can not connect to remote server with port=" + std::to_string(node->port) + " and address=" + node->address + ");");

      // throw an exception
      throw std::runtime_error(errMsg);
    }

    // we connected
    break;
  }

  // make a block to send the request
  const UseTemporaryAllocationBlock tempBlock{1024};

  // make the request
  Handle<StoGetPageRequest> pageRequest = makeObject<StoGetPageRequest>(databaseName, setName, page);

  // send the object
  if (!comm->sendObject(pageRequest, errMsg)) {

    // yeah something happened
    logger->error(errMsg);
    logger->error("Not able to send request to server.\n");

    // throw an exception
    throw std::runtime_error(errMsg);
  }

  // get the response and process it
  bool success;
  Handle<StoGetPageResult> result = comm->getNextObject<StoGetPageResult> (success, errMsg);

  // did we get a response
  if(result == nullptr) {

    // throw an exception
    throw std::runtime_error(errMsg);
  }

  // do we have a next page
  if(!result->hasPage) {
    return std::make_pair(nullptr, result->size);
  }

  // grab a page
  auto pageHandle = getFunctionalityPtr<PDBBufferManagerInterface>()->getPage(result->size);

  // get the bytes
  auto readSize = RequestFactory::waitForBytes(logger, comm, (char*) pageHandle->getBytes(), result->size, errMsg);

  // did we fail to read it...
  if(readSize == -1) {
    return std::make_pair(nullptr, result->size);
  }

  // return it
  return std::make_pair(pageHandle, result->size);
}
