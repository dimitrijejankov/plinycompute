//
// Created by dimitrije on 5/7/19.
//

#include <ComputePlan.h>
#include <PDBCatalogClient.h>
#include <physicalAlgorithms/PDBShuffleForJoinAlgorithm.h>
#include <ExJob.h>
#include <PDBStorageManagerBackend.h>
#include <PDBPageNetworkSender.h>
#include <ShuffleJoinProcessor.h>
#include <PDBPageSelfReceiver.h>
#include <PDBShuffleForJoinStage.h>
#include <PDBShuffleForJoinState.h>
#include <GenericWork.h>
#include <memory>

pdb::PDBShuffleForJoinAlgorithm::PDBShuffleForJoinAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                                                            const AtomicComputationPtr &finalAtomicComputation,
                                                            const pdb::Handle<pdb::PDBSinkPageSetSpec> &intermediate,
                                                            const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                                                            const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                                            const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize)
    : PDBPhysicalAlgorithm(primarySource, finalAtomicComputation, sink, secondarySources, setsToMaterialize),
      intermediate(intermediate) {

}

pdb::PDBPhysicalAlgorithmType pdb::PDBShuffleForJoinAlgorithm::getAlgorithmType() {
  return ShuffleForJoin;
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::PDBShuffleForJoinAlgorithm::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {

  // init the state
  auto state = std::make_shared<PDBShuffleForJoinState>();

  // init the logger for this algorithm
  state->logger = make_shared<PDBLogger>("PDBShuffleForJoinAlgorithm" + std::to_string(job->computationID));

  // return the state
  return state;
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::PDBShuffleForJoinAlgorithm::getNextStage() {

  // we are done if we already served a stage
  if(currentStage == 1) {
    return nullptr;
  }

  // go to the next stage
  currentStage++;

  // return the shuffle join pipe stage
  return { pdb::make_shared<PDBShuffleForJoinStage>(*sink,
                                                    sources,
                                                    finalTupleSet,
                                                    *secondarySources,
                                                    *setsToMaterialize,
                                                    *intermediate) };
}

int32_t pdb::PDBShuffleForJoinAlgorithm::numStages() const {
  return 1;
}
