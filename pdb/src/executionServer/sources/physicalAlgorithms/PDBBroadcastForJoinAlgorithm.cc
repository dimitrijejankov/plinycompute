//
// Created by dimitrije on 5/7/19.
//

#include <physicalAlgorithms/PDBBroadcastForJoinAlgorithm.h>

pdb::PDBBroadcastForJoinAlgorithm::PDBBroadcastForJoinAlgorithm(const std::string &firstTupleSet,
                                                                const std::string &finalTupleSet,
                                                                const pdb::Handle<pdb::PDBSourcePageSetSpec> &source,
                                                                const pdb::Handle<pdb::PDBSinkPageSetSpec> &intermediate,
                                                                const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                                                                const pdb::Handle<pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>>> &secondarySources,
                                                                const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize,
                                                                const bool swapLHSandRHS)
                                                                : PDBPhysicalAlgorithm(firstTupleSet, finalTupleSet, source, sink, secondarySources, setsToMaterialize, swapLHSandRHS),
                                                                  intermediate(intermediate) {


}

pdb::PDBPhysicalAlgorithmType pdb::PDBBroadcastForJoinAlgorithm::getAlgorithmType() {
  return BroadcastForJoin;
}

bool pdb::PDBBroadcastForJoinAlgorithm::run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  return true;
}

bool pdb::PDBBroadcastForJoinAlgorithm::setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) {
  return true;
}