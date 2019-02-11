//
// Created by dimitrije on 2/9/19.
//

#ifndef PDB_STORAGEMANAGERFRONTEND_H
#define PDB_STORAGEMANAGERFRONTEND_H

#include <ServerFunctionality.h>
#include <PDBSet.h>
#include <PDBPageCompare.h>
#include <mutex>

namespace pdb {

class StorageManagerFrontend : public ServerFunctionality {

public:

  /**
   * Initialize the storage manager frontend
   */
  void init() override;

  void registerHandlers(PDBServer &forMe) override;


 private:

  /**
   * The logger
   */
  PDBLoggerPtr logger;

  /**
   * The last page for a particular set
   */
  map <PDBSetPtr, size_t, PDBSetCompare> lastPages;


  std::mutex m;
};

}


#endif //PDB_STORAGEMANAGERFRONTEND_H
