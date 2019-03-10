//
// Created by dimitrije on 3/8/19.
//

#ifndef PDB_EXECUTIONSERVERBACKEND_H
#define PDB_EXECUTIONSERVERBACKEND_H

#include <ServerFunctionality.h>

namespace pdb {

class ExecutionServerBackend : public ServerFunctionality {

  void registerHandlers(PDBServer &forMe) override;

  void init() override;

};

}

#endif //PDB_EXECUTIONSERVERBACKEND_H
