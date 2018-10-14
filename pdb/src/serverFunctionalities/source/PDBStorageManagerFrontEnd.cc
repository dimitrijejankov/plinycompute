#include <PDBStorageManagerFrontEnd.h>

#include <PagedRequestHandler.h>
#include <StoGetPage.h>

void pdb::PDBStorageManagerFrontEnd::registerHandlers(pdb::PDBServer &forMe) {
  forMe.registerHandler(
      StoGetPage_TYPEID,
      make_shared<pdb::PagedRequestHandler<StoGetPage>>(
      [&](Handle<StoGetPage> request, PDBCommunicatorPtr sendUsingMe) {



        return make_pair(true, "");
      }));
}

PDBStorageManagerInterfacePtr pdb::PDBStorageManagerFrontEnd::getBackEnd() {
  return PDBStorageManagerInterfacePtr();
}
