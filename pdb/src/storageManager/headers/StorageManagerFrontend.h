//
// Created by dimitrije on 2/9/19.
//

#ifndef PDB_STORAGEMANAGERFRONTEND_H
#define PDB_STORAGEMANAGERFRONTEND_H

#include <ServerFunctionality.h>

namespace pdb {

class StorageManagerFrontend : public ServerFunctionality {

public:
  
  void registerHandlers(PDBServer &forMe) override;

};

}


#endif //PDB_STORAGEMANAGERFRONTEND_H
