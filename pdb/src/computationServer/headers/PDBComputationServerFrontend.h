//
// Created by dimitrije on 2/19/19.
//

#ifndef PDB_COMPUTATIONSERVER_H
#define PDB_COMPUTATIONSERVER_H

#include <ServerFunctionality.h>

namespace pdb {

class PDBComputationServerFrontend : public ServerFunctionality {

public:

  void registerHandlers(PDBServer &forMe) override;

};


}


#endif //PDB_COMPUTATIONSERVER_H
