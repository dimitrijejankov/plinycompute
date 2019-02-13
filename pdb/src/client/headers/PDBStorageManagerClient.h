//
// Created by dimitrije on 2/13/19.
//

#ifndef PDB_PDBSTORAGEMANAGERCLIENT_H
#define PDB_PDBSTORAGEMANAGERCLIENT_H

#include <ServerFunctionality.h>
#include <PDBPage.h>
#include <PDBStorageIterator.h>

namespace pdb {

class PDBStorageManagerClient : public ServerFunctionality {

public:

  /**
   * Registers the handles needed for the server functionality
   * @param forMe
   */
  void registerHandlers(PDBServer &forMe) override {};

  /**
   * Returns an iterator that can fetch records from the storage
   * @param set - the set want to grab the iterator for
   * @return the iterator
   */
  template <class T>
  PDBStorageIteratorPtr<T> getIterator(PDBSetPtr set, std::string &error, bool &success);

private:

  /**
   * The port of the manager
   */
  int port = -1;

  /**
   * The address of the manager
   */
  std::string address;

  /**
   * The logger of the client
   */
  PDBLoggerPtr logger;

};


}

#include "PDBStorageManagerClientTemplate.cc"

#endif //PDB_PDBSTORAGEMANAGERCLIENT_H
