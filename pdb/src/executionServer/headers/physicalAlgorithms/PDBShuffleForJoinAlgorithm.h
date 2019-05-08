#pragma once

#include "PDBPhysicalAlgorithm.h"

namespace pdb {

class PDBShuffleForJoinAlgorithm : public PDBPhysicalAlgorithm {
public:

  PDBShuffleForJoinAlgorithm(const std::string &firstTupleSet,
                             const std::string &finalTupleSet,
                             const pdb::Handle<PDBSourcePageSetSpec> &source,
                             const pdb::Handle<PDBSinkPageSetSpec> &sink,
                             const pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &secondarySources);

};

}