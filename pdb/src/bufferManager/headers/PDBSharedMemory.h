//
// Created by dimitrije on 10/14/18.
//

#ifndef PDB_PDBSTORAGE_H
#define PDB_PDBSTORAGE_H

#include <memory>

struct PDBSharedMemory {

  // pointer to the shared memory
  void *memory;

  // the page size
  size_t pageSize;

  // the number of pages of RAM in the buffer
  size_t numPages;

};

#endif //PDB_PDBSTORAGE_H
