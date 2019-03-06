//
// Created by dimitrije on 3/5/19.
//

#ifndef PDB_ABSTRATCTPAGESET_H
#define PDB_ABSTRATCTPAGESET_H

#include <PDBPageHandle.h>

namespace pdb {

class PDBAbstractPageSet;
using PDBAbstractPageSetPtr = std::shared_ptr<PDBAbstractPageSet>;

class PDBAbstractPageSet {
public:

  virtual PDBPageHandle getNextPage(size_t workerID) = 0;

  virtual PDBPageHandle getNewPage() = 0;

};

}

#endif //PDB_ABSTRATCTPAGESET_H
