//
// Created by dimitrije on 4/9/19.
//

#ifndef PDB_SILLYWRITE_H
#define PDB_SILLYWRITE_H

#include <SetWriter.h>

namespace pdb {

class SillyWrite : public SetWriter <Handle <String>> {

public:

  ENABLE_DEEP_COPY

};

}


#endif //PDB_SILLYWRITE_H
