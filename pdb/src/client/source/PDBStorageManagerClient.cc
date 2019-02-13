//
// Created by dimitrije on 2/13/19.
//

#include <PDBStorageManagerClient.h>
#include "PDBStorageManagerClient.h"
#include <HeapRequest.h>

pdb::PDBPagePtr pdb::PDBStorageManagerClient::requestPage(pdb::PDBSetPtr set, size_t page, std::string &error, bool &success) {
  return pdb::PDBPagePtr();
}

size_t pdb::PDBStorageManagerClient::getNumPages(pdb::PDBSetPtr set, std::string &error, bool &success) {
//  return RequestFactory::heapRequest<ShutDown, SimpleRequestResult, bool>(logger, w->port, w->address, false, 1024,
//                                                                          [&](Handle<SimpleRequestResult> result) {
//
//                                                                            // do we have a result
//                                                                            if(result == nullptr) {
//
//                                                                              errMsg = "Error getting type name: got nothing back from catalog";
//                                                                              return false;
//                                                                            }
//
//                                                                            // did we succeed
//                                                                            if (!result->getRes().first) {
//
//                                                                              errMsg = "Error shutting down server: " + result->getRes().second;
//                                                                              logger->error("Error shutting down server: " + result->getRes().second);
//
//                                                                              return false;
//                                                                            }
//
//                                                                            // we succeeded
//                                                                            return true;
//                                                                          });

  return 0;
}
