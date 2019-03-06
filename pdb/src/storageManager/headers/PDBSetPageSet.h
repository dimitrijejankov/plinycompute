//
// Created by dimitrije on 3/5/19.
//

#ifndef PDB_PDBSETPAGESET_H
#define PDB_PDBSETPAGESET_H

#include "PDBAbstractPageSet.h"
#include <PDBBufferManagerInterface.h>
#include <vector>

namespace pdb {

class PDBStorageManagerBackend;

class PDBSetPageSet : public PDBAbstractPageSet {
public:

  PDBSetPageSet(const std::string &db, const std::string &set, size_t numPages, PDBBufferManagerInterfacePtr bufferManager);

  PDBPageHandle getNextPage(size_t workerID) override;

  PDBPageHandle getNewPage() override;

private:

  // current page, it is thread safe to update it
  atomic_uint64_t curPage;

  // last page, it is thread safe
  atomic_uint64_t lastPage;

  // the database the set this page set corresponds to belongs
  std::string db;

  // the name of the set this page set corresponds to
  std::string set;

  // the page handles
  std::vector<PDBPageHandle> pages;

  // the buffer manager to get the pages
  PDBBufferManagerInterfacePtr bufferManager;
};

}

#endif //PDB_PDBSETPAGESET_H
