//
// Created by dimitrije on 4/9/19.
//

#ifndef PDB_SILLYREADOFB_H
#define PDB_SILLYREADOFB_H

#include <StringIntPair.h>
#include <SetScanner.h>

namespace pdb {

class SillyReadOfB : public SetScanner<StringIntPair> {

  ENABLE_DEEP_COPY

};

}

#endif //PDB_SILLYREADOFB_H
