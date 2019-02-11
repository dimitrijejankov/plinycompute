//
// Created by dimitrije on 2/11/19.
//

#ifndef PDB_STORAGEMANAGERBACKEND_H
#define PDB_STORAGEMANAGERBACKEND_H

#include <ServerFunctionality.h>

namespace pdb {

class StorageManagerBackend : public ServerFunctionality {

public:

  void registerHandlers(PDBServer &forMe) override;
};

}



#endif //PDB_STORAGEMANAGERBACKEND_H
