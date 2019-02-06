//
// Created by dimitrije on 2/5/19.
//

#ifndef PDB_DISPATCHERSERVERBACKEND_H
#define PDB_DISPATCHERSERVERBACKEND_H

#include <PDBServer.h>

namespace pdb {

class DispatcherServerBackend : public ServerFunctionality  {

public:

  void init() override {}

  void registerHandlers(PDBServer &forMe) override {

  }

};

}

#endif //PDB_DISPATCHERSERVERBACKEND_H
