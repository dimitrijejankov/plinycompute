//
// Created by dimitrije on 2/5/19.
//

#ifndef PDB_DISPATCHERSERVERFRONTEND_H
#define PDB_DISPATCHERSERVERFRONTEND_H

#include <DataTypes.h>
#include <ServerFunctionality.h>

namespace pdb {

class DispatcherServerFrontend : public ServerFunctionality  {

public:

  void init() override {

  }

  void registerHandlers(PDBServer &forMe) override {

  }

};

}


#endif //PDB_DISPATCHERSERVERFRONTEND_H
