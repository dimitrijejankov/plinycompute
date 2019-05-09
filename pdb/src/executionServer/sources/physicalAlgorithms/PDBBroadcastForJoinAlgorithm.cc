//
// Created by dimitrije on 5/7/19.
//

#include <physicalAlgorithms/PDBBroadcastForJoinAlgorithm.h>

pdb::PDBBroadcastForJoinAlgorithm::PDBBroadcastForJoinAlgorithm(const std::string &firstTupleSet,
                                                                const std::string &finalTupleSet,
                                                                const pdb::Handle<pdb::PDBSourcePageSetSpec> &source,
                                                                const pdb::Handle<pdb::PDBSinkPageSetSpec> &intermediate,
                                                                const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                                                                const pdb::Handle<pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>>> &secondarySources)
                                                                : PDBPhysicalAlgorithm(firstTupleSet, finalTupleSet, source, sink, secondarySources),
                                                                  intermediate(intermediate) {


}

pdb::PDBPhysicalAlgorithmType pdb::PDBBroadcastForJoinAlgorithm::getAlgorithmType() {
  return BroadcastForJoin;
}