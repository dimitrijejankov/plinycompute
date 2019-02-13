//
// Created by dimitrije on 2/13/19.
//

#ifndef PDB_PDBSTORAGEMANAGERCLIENT_H
#define PDB_PDBSTORAGEMANAGERCLIENT_H

#include <ServerFunctionality.h>
#include <PDBPage.h>

namespace pdb {

class PDBStorageManagerClient : public ServerFunctionality {

public:

  /**
   * Registers the handles needed for the server functionality
   * @param forMe
   */
  void registerHandlers(PDBServer &forMe) override {};

  /**
   *
   * @param set
   * @param page
   * @return
   */
  PDBPagePtr requestPage(PDBSetPtr set, size_t page, std::string &error, bool &success);

  /**
   *
   * @param set
   * @return
   */
  size_t getNumPages(PDBSetPtr set, std::string &error, bool &success);

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


#endif //PDB_PDBSTORAGEMANAGERCLIENT_H
