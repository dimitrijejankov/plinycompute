#include <PDBJoin3AggAsyncStage.h>
#include <PDBJoin3AlgorithmState.h>
#include <PDBJoinAggregationState.h>
#include <ExJob.h>
#include <ComputePlan.h>
#include <AtomicComputationClasses.h>
#include <Join8SideSender.h>
#include <GenericWork.h>

bool pdb::PDBJoin3AggAsyncStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                       const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                       const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                       const std::string &error) {


  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin3AlgorithmState>(state);

  //



  return true;
}

bool pdb::PDBJoin3AggAsyncStage::run(const pdb::Handle<pdb::ExJob> &job,
                                     const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                     const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                     const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin3AlgorithmState>(state);


  return true;
}

void pdb::PDBJoin3AggAsyncStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  // cast the state
  auto s = dynamic_pointer_cast<pdb::PDBJoin3AlgorithmState>(state);

}
