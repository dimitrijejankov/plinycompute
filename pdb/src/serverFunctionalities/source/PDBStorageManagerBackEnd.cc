//
// Created by dimitrije on 10/12/18.
//

#include <PDBStorageManagerBackEnd.h>

pdb::PDBPageHandle pdb::PDBStorageManagerBackEnd::getPage(pdb::PDBSetPtr whichSet, uint64_t i) {
  return pdb::PDBPageHandle();
}

pdb::PDBPageHandle pdb::PDBStorageManagerBackEnd::getPage() {
  return pdb::PDBPageHandle();
}

pdb::PDBPageHandle pdb::PDBStorageManagerBackEnd::getPage(size_t minBytes) {
  return pdb::PDBPageHandle();
}

size_t pdb::PDBStorageManagerBackEnd::getPageSize() {
  return 0;
}

void pdb::PDBStorageManagerBackEnd::registerHandlers(pdb::PDBServer &forMe) {

}

void pdb::PDBStorageManagerBackEnd::freeAnonymousPage(pdb::PDBPagePtr me) {

}

void pdb::PDBStorageManagerBackEnd::downToZeroReferences(pdb::PDBPagePtr me) {

}

void pdb::PDBStorageManagerBackEnd::freezeSize(pdb::PDBPagePtr me, size_t numBytes) {

}

void pdb::PDBStorageManagerBackEnd::unpin(pdb::PDBPagePtr me) {

}

void pdb::PDBStorageManagerBackEnd::repin(pdb::PDBPagePtr me) {

}
