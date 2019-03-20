//
// Created by dimitrije on 3/20/19.
//

#include "PDBAggregationPipeAlgorithm.h"

pdb::PDBAggregationPipeAlgorithm::PDBAggregationPipeAlgorithm(const pdb::Handle<pdb::PDBSourcePageSetSpec> &source,
                                                              const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                                                              const pdb::Handle<pdb::Vector<pdb::PDBSourcePageSetSpec>> &secondarySources) : PDBPhysicalAlgorithm(source, sink, secondarySources) {}
