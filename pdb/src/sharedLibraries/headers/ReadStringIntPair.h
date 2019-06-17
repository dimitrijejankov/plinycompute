//
// Created by dimitrije on 4/9/19.
//

#ifndef PDB_ReadStringIntPair_H
#define PDB_ReadStringIntPair_H

#include <SetScanner.h>
#include <StringIntPair.h>

namespace pdb {

class ReadStringIntPair : public SetScanner <StringIntPair> {
public:

  ReadStringIntPair() = default;

  ReadStringIntPair(const std::string &db, const std::string &set) : SetScanner(db, set) {}

  ENABLE_DEEP_COPY
};



}

#endif //PDB_ReadStringIntPair_H
