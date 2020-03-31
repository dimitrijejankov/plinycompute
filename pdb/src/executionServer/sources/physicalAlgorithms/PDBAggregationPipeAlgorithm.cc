#include "PDBAggregationPipeAlgorithm.h"
#include "PDBAggregationPipeState.h"
#include "PDBAggregationPipeStage.h"
#include <SourceSetArg.h>
#include <PDBCatalogClient.h>
#include "ComputePlan.h"
#include "ExJob.h"
#include "PDBStorageManagerBackend.h"
#include "GenericWork.h"

pdb::PDBAggregationPipeAlgorithm::PDBAggregationPipeAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                                                              const AtomicComputationPtr &finalAtomicComputation,
                                                              const Handle<PDBSinkPageSetSpec> &hashedToSend,
                                                              const Handle<PDBSourcePageSetSpec> &hashedToRecv,
                                                              const Handle<PDBSinkPageSetSpec> &sink,
                                                              const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                                              const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize)

    : PDBPhysicalAlgorithm(primarySource, finalAtomicComputation, sink, secondarySources, setsToMaterialize),
      hashedToSend(hashedToSend),
      hashedToRecv(hashedToRecv) {}

pdb::PDBPhysicalAlgorithmStatePtr pdb::PDBAggregationPipeAlgorithm::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {

  // init the state
  auto state = std::make_shared<PDBAggregationPipeState>();

  // init the logger for this algorithm
  state->logger = make_shared<PDBLogger>("PDBAggregationPipeAlgorithm_" + std::to_string(job->computationID));

  // return the state
  return state;
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::PDBAggregationPipeAlgorithm::getNextStage(const PDBPhysicalAlgorithmStatePtr &state) {

  // we are done if we already served a stage
  if(currentStage == 1) {
    return nullptr;
  }

  // go to the next stage
  currentStage++;

  // return the aggregation pipe stage
  return std::make_shared<PDBAggregationPipeStage>(*sink,
                                                   sources,
                                                   finalTupleSet,
                                                   *secondarySources,
                                                   *setsToMaterialize,
                                                   *hashedToSend,
                                                   *hashedToRecv);
}

int32_t pdb::PDBAggregationPipeAlgorithm::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::PDBAggregationPipeAlgorithm::getAlgorithmType() {
  return DistributedAggregation;
}

pdb::PDBCatalogSetContainerType pdb::PDBAggregationPipeAlgorithm::getOutputContainerType() {
  return PDBCatalogSetContainerType::PDB_CATALOG_SET_MAP_CONTAINER;
}