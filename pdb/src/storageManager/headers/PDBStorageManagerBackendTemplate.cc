
#ifndef PDB_PDBSTORAGEMANAGERBACKENDTEMPLATE_H
#define PDB_PDBSTORAGEMANAGERBACKENDTEMPLATE_H

#include <ServerFunctionality.h>
#include <StoStoreOnPageRequest.h>
#include <PDBStorageManagerBackend.h>
#include <PDBBufferManagerBackEnd.h>

template <class Communicator>
std::pair<bool, std::string> pdb::PDBStorageManagerBackend::handleStoreOnPage(const pdb::Handle<pdb::StoStoreOnPageRequest> &request,
                                                                              std::shared_ptr<Communicator> &sendUsingMe) {

  /// 1. Grab a page and decompress the forwarded page

  // grab the buffer manager
  auto bufferManager = std::dynamic_pointer_cast<pdb::PDBBufferManagerBackEndImpl>(this->getFunctionalityPtr<PDBBufferManagerInterface>());

  // grab the forwarded page
  auto inPage = bufferManager->expectPage(sendUsingMe);

  // check the uncompressed size
  size_t uncompressedSize = 0;
  snappy::GetUncompressedLength((char*) inPage->getBytes(), request->compressedSize, &uncompressedSize);

  // grab the page
  auto outPage = bufferManager->getPage(make_shared<pdb::PDBSet>(request->databaseName, request->setName), request->page);

  // uncompress and copy to page
  snappy::RawUncompress((char*) inPage->getBytes(), request->compressedSize, (char*) outPage->getBytes());

  // freeze the size
  outPage->freezeSize(uncompressedSize);

  /// 2. Send the response that we are done

  // create an allocation block to hold the response
  string error;
  pdb::Handle<pdb::SimpleRequestResult> simpleResponse = pdb::makeObject<pdb::SimpleRequestResult>(true, error);

  // sends result to requester
  sendUsingMe->sendObject(simpleResponse, error);

  // finish
  return make_pair(true, error);
}


#endif //PDB_PDBSTORAGEMANAGERBACKENDTEMPLATE_H
