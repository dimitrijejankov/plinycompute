#include <physicalAlgorithms/PDBBroadcastForJoinAlgorithm.h>
#include <physicalAlgorithms/PDBBroadcastForJoinState.h>
#include <physicalAlgorithms/PDBBroadcastForJoinStage.h>

pdb::PDBBroadcastForJoinAlgorithm::PDBBroadcastForJoinAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                                                                const AtomicComputationPtr &finalAtomicComputation,
                                                                const pdb::Handle<pdb::PDBSinkPageSetSpec> &hashedToSend,
                                                                const pdb::Handle<pdb::PDBSourcePageSetSpec> &hashedToRecv,
                                                                const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                                                                const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                                                const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize)
    :
    PDBPhysicalAlgorithm(primarySource,
                         finalAtomicComputation,
                         sink,
                         secondarySources,
                         setsToMaterialize),
    hashedToSend(hashedToSend),
    hashedToRecv(hashedToRecv) {
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::PDBBroadcastForJoinAlgorithm::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {

  // init the state
  auto state = std::make_shared<PDBBroadcastForJoinState>();

  // init the logger for this algorithm
  state->logger = make_shared<PDBLogger>("PDBBroadcastForJoinAlgorithm_" + std::to_string(job->computationID));

  // return the state
  return state;
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::PDBBroadcastForJoinAlgorithm::getNextStage(const PDBPhysicalAlgorithmStatePtr &state) {

  // we are done if we already served a stage
  if(currentStage == 1) {
    return nullptr;
  }

  // go to the next stage
  currentStage++;

  // return the broadcast pipe stage
  return std::make_shared<PDBBroadcastForJoinStage>(*sink,
                                                    sources,
                                                    finalTupleSet,
                                                    *secondarySources,
                                                    *setsToMaterialize,
                                                    *hashedToSend,
                                                    *hashedToRecv);
}

int32_t pdb::PDBBroadcastForJoinAlgorithm::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::PDBBroadcastForJoinAlgorithm::getAlgorithmType() {
  return BroadcastForJoin;
}
