//
// Created by dimitrije on 2/11/19.
//

#include <PDBBufferManagerBackEnd.h>
#include <SharedEmployee.h>
#include "PDBStorageManagerBackend.h"
#include "HeapRequestHandler.h"
#include "StoStoreOnPageRequest.h"

void pdb::PDBStorageManagerBackend::registerHandlers(PDBServer &forMe) {

  forMe.registerHandler(
      StoStoreOnPageRequest_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::StoStoreOnPageRequest>>(
          [&](Handle<pdb::StoStoreOnPageRequest> request, PDBCommunicatorPtr sendUsingMe) {

            /// 1. Grab a page and decompress the forwarded page

            // grab the buffer manager
            auto bufferManager = std::dynamic_pointer_cast<PDBStorageManagerBackEndImpl>(getFunctionalityPtr<PDBBufferManagerInterface>());

            // grab the forwarded page
            auto inPage = bufferManager->expectPage(sendUsingMe);

            // check the uncompressed size
            size_t uncompressedSize = 0;
            snappy::GetUncompressedLength((char*) inPage->getBytes(), request->compressedSize, &uncompressedSize);

            // grab the page
            auto outPage = bufferManager->getPage(std::make_shared<PDBSet>(request->setName, request->databaseName), request->page);

            // uncompress and copy to page
            snappy::RawUncompress((char*) inPage->getBytes(), request->compressedSize, (char*) outPage->getBytes());

            // freeze the size
            outPage->freezeSize(uncompressedSize);

            /// 2. Send the response that we are done

            // create an allocation block to hold the response
            std::string error;
            Handle<SimpleRequestResult> simpleResponse = makeObject<SimpleRequestResult>(true, error);

            // sends result to requester
            sendUsingMe->sendObject(simpleResponse, error);

            // finish
            return std::make_pair(true, error);
          }));

}
