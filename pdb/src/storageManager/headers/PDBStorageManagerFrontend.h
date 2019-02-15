//
// Created by dimitrije on 2/9/19.
//

#ifndef PDB_STORAGEMANAGERFRONTEND_H
#define PDB_STORAGEMANAGERFRONTEND_H

#include <mutex>

#include <PDBSet.h>
#include <PDBPageCompare.h>
#include <PDBCatalogNode.h>
#include <ServerFunctionality.h>
#include <PDBPageHandle.h>

namespace pdb {

class PDBStorageManagerFrontend : public ServerFunctionality {

public:

  virtual ~PDBStorageManagerFrontend();

  /**
   * Initialize the storage manager frontend
   */
  void init() override;

  void registerHandlers(PDBServer &forMe) override;

  std::pair<PDBPageHandle, size_t> requestPage(const PDBCatalogNodePtr& node, const std::string &databaseName, const std::string &setName, uint64_t page);

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
