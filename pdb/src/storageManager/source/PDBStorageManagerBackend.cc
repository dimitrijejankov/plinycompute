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
            return handleStoreOnPage(request, sendUsingMe);
      }));
}