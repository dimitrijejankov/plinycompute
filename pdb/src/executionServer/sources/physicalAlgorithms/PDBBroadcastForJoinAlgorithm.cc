//
// Created by dimitrije on 5/7/19.
//

#include <physicalAlgorithms/PDBBroadcastForJoinAlgorithm.h>

pdb::PDBBroadcastForJoinAlgorithm::PDBBroadcastForJoinAlgorithm(const std::string &firstTupleSet,
                                                                const std::string &finalTupleSet,
                                                                const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                                const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                                                const pdb::Handle<pdb::Vector<pdb::Handle<
                                                                    PDBSourcePageSetSpec>>> &secondarySources) : PDBPhysicalAlgorithm(firstTupleSet, finalTupleSet, source, sink, secondarySources) {

}

