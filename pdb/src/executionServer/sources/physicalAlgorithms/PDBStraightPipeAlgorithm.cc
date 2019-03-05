//
// Created by dimitrije on 2/25/19.
//

#include <PDBVector.h>
#include "physicalAlgorithms/PDBStraightPipeAlgorithm.h"

pdb::PDBStraightPipeAlgorithm::PDBStraightPipeAlgorithm(const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                        const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                                        const pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> &secondarySources)
                                                        : PDBPhysicalAlgorithm(source, sink, secondarySources) {}


void pdb::PDBStraightPipeAlgorithm::setup() {

}

void pdb::PDBStraightPipeAlgorithm::run() {

}

pdb::PDBPhysicalAlgorithmType pdb::PDBStraightPipeAlgorithm::getAlgorithmType() {
  return StraightPipe;
}
