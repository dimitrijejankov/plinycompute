#include <PDBStorageManagerFrontEnd.h>

#include <PagedRequestHandler.h>
#include <StoGetPage.h>
#include <PDBStorageManagerBackEnd.h>

void pdb::PDBStorageManagerFrontEnd::registerHandlers(pdb::PDBServer &forMe) {
  forMe.registerHandler(
      StoGetPage_TYPEID,
      make_shared<pdb::PagedRequestHandler<StoGetPage>>(
      [&](Handle<StoGetPage> request, PDBCommunicatorPtr sendUsingMe) {



        return make_pair(true, "");
      }));
}

pdb::PDBStorageManagerInterfacePtr pdb::PDBStorageManagerFrontEnd::getBackEnd() {

  // init the backend storage manager with the shared memory
  return std::make_shared<PDBStorageManagerBackEnd>(sharedMemory);
}
