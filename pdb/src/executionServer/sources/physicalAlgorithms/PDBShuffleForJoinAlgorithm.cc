//
// Created by dimitrije on 5/7/19.
//

#include <physicalAlgorithms/PDBShuffleForJoinAlgorithm.h>

pdb::PDBShuffleForJoinAlgorithm::PDBShuffleForJoinAlgorithm(const std::string &firstTupleSet,
                                                            const std::string &finalTupleSet,
                                                            const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                            const pdb::Handle<PDBSinkPageSetSpec> &intermediate,
                                                            const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                                            const pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &secondarySources)
                                                            : PDBPhysicalAlgorithm(firstTupleSet, finalTupleSet, source, sink, secondarySources),
                                                              intermediate(intermediate) {

}

pdb::PDBPhysicalAlgorithmType pdb::PDBShuffleForJoinAlgorithm::getAlgorithmType() {
  return ShuffleForJoin;
}

bool pdb::PDBShuffleForJoinAlgorithm::run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  return true;
}

bool pdb::PDBShuffleForJoinAlgorithm::setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) {
  return true;
}