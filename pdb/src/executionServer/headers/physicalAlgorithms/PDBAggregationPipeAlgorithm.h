//
// Created by dimitrije on 3/20/19.
//

#ifndef PDB_PDBAGGREGATIONPIPEALGORITHM_H
#define PDB_PDBAGGREGATIONPIPEALGORITHM_H

// PRELOAD %PDBAggregationPipeAlgorithm%

#include "PDBPhysicalAlgorithm.h"

namespace pdb {

class PDBAggregationPipeAlgorithm : public PDBPhysicalAlgorithm {
public:

  PDBAggregationPipeAlgorithm() = default;

  PDBAggregationPipeAlgorithm(const Handle<PDBSourcePageSetSpec> &source,
                              const Handle<PDBSinkPageSetSpec> &sink,
                              const Handle<Vector<PDBSourcePageSetSpec>> &secondarySources);

};

}


#endif //PDB_PDBAGGREGATIONPIPEALGORITHM_H
