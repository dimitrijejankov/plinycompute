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

class PDBStorageManagerFrontend : public ServerFunctionality {

public:

  virtual ~PDBStorageManagerFrontend();

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
