//
// Created by dimitrije on 3/20/19.
//

#include "PDBAggregationPipeAlgorithm.h"

pdb::PDBAggregationPipeAlgorithm::PDBAggregationPipeAlgorithm(const pdb::Handle<pdb::PDBSourcePageSetSpec> &source,
                                                              const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                                                              const pdb::Handle<pdb::Vector<pdb::PDBSourcePageSetSpec>> &secondarySources) : PDBPhysicalAlgorithm(source, sink, secondarySources) {}

bool pdb::PDBAggregationPipeAlgorithm::setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                             Handle<pdb::ExJob> &job,
                                             const std::string &error) {

}

bool pdb::PDBAggregationPipeAlgorithm::run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

}

pdb::PDBPhysicalAlgorithmType pdb::PDBAggregationPipeAlgorithm::getAlgorithmType() {
  return DistributedAggregation;
}