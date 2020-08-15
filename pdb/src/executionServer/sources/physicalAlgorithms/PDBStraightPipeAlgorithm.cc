//
// Created by dimitrije on 2/25/19.
//

#include <PDBVector.h>
#include <ComputePlan.h>
#include <GenericWork.h>
#include <PDBCatalogClient.h>
#include <PDBStraightPipeState.h>
#include <physicalAlgorithms/PDBStraightPipeAlgorithm.h>
#include <physicalAlgorithms/PDBPhysicalAlgorithmStage.h>
#include <physicalAlgorithms/PDBStraightPipeStage.h>
#include <boost/filesystem/path.hpp>
#include "ExJob.h"

pdb::PDBStraightPipeAlgorithm::PDBStraightPipeAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                                                        const AtomicComputationPtr &finalAtomicComputation,
                                                        const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                                        const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                                        const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize)
    : PDBPhysicalAlgorithm(primarySource, finalAtomicComputation, sink, secondarySources, setsToMaterialize) {}

pdb::PDBPhysicalAlgorithmType pdb::PDBStraightPipeAlgorithm::getAlgorithmType() {
  return StraightPipe;
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::PDBStraightPipeAlgorithm::getInitialState(const Handle<pdb::ExJob> &job,
                                                                                 NodeConfigPtr config) const {

  // init the state
  auto state = std::make_shared<PDBStraightPipeState>();

  // init the logger for this algorithm
  state->logger = make_shared<PDBLogger>((boost::filesystem::path(config->rootDirectory) / "logs").string(),
                                         "PDBStraightPipeAlgorithm" + std::to_string(job->computationID));

  // return the state
  return state;
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::PDBStraightPipeAlgorithm::getNextStage(const PDBPhysicalAlgorithmStatePtr &state) {

  // we are done if we already served a stage
  if(currentStage == 1) {
    return nullptr;
  }

  // go to the next stage
  currentStage++;

  // return the straight pipe stage
  return { std::make_shared<PDBStraightPipeStage>(*sink,
                                                 sources,
                                                 finalTupleSet,
                                                 *secondarySources,
                                                 *setsToMaterialize) };
}

int32_t pdb::PDBStraightPipeAlgorithm::numStages() const {
  return 1;
}

pdb::PDBCatalogSetContainerType pdb::PDBStraightPipeAlgorithm::getOutputContainerType() {
  return PDBCatalogSetContainerType::PDB_CATALOG_SET_VECTOR_CONTAINER;
}