//
// Created by dimitrije on 4/9/19.
//

#ifndef PDB_SILLYREADOFA_H
#define PDB_SILLYREADOFA_H

#include <SetScanner.h>

namespace pdb {

class ReadInt : public SetScanner <int> {
public:

  ReadInt() = default;

  ReadInt(const std::string &db, const std::string &set) : SetScanner(db, set) {}

  ENABLE_DEEP_COPY

};

}

#endif //PDB_SILLYREADOFA_H
