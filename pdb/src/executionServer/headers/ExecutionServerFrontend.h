//
// Created by dimitrije on 3/4/19.
//

#ifndef PDB_EXECUTIONSERVER_H
#define PDB_EXECUTIONSERVER_H

#include <ServerFunctionality.h>

namespace pdb {

class ExecutionServerFrontend : public ServerFunctionality {

public:

  void registerHandlers(PDBServer &forMe) override;

  void init() override;

  /**
   * The logger
   */
  PDBLoggerPtr logger;

};

}

#endif //PDB_EXECUTIONSERVER_H
