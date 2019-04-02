//
// Created by dimitrije on 3/20/19.
//

#include <physicalAlgorithms/PDBAggregationPipeAlgorithm.h>

#include "PDBAggregationPipeAlgorithm.h"

pdb::PDBAggregationPipeAlgorithm::PDBAggregationPipeAlgorithm(const std::string &firstTupleSet,
                                                              const std::string &finalTupleSet,
                                                              const pdb::Handle<pdb::PDBSourcePageSetSpec> &source,
                                                              const pdb::Handle<pdb::PDBSinkPageSetSpec> &hashedToSend,
                                                              const pdb::Handle<pdb::PDBSourcePageSetSpec> &hashedToRecv,
                                                              const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                                                              const pdb::Handle<pdb::Vector<pdb::PDBSourcePageSetSpec>> &secondarySources)
    : PDBPhysicalAlgorithm(firstTupleSet, finalTupleSet, source, sink, secondarySources), hashedToSend(hashedToSend), hashedToRecv(hashedToRecv) {}


bool pdb::PDBAggregationPipeAlgorithm::setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) {
  return true;
}

bool pdb::PDBAggregationPipeAlgorithm::run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  return true;
}