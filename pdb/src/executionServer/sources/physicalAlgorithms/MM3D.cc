#include <MM3D.h>
#include <MM3DMultiplyStage.h>
#include <MM3DShuffleStage.h>
#include <ExJob.h>
#include <physicalAlgorithms/MM3DState.h>

pdb::MM3D::MM3D(int32_t n, int32_t num_nodes, int32_t num_threads) : n(n),
                                                                     num_nodes(num_nodes),
                                                                     num_threads(num_threads) {
  // init the sets to materialize
  setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::MM3D::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {

  auto s = std::make_shared<MM3DState>();
  s->logger = make_shared<PDBLogger>("PDBPhysicalAlgorithm_" + std::to_string(job->computationID));

  return s;
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::MM3D::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {

  // we are done if we already served a stage
  if (currentStage == 2) {
    return nullptr;
  }

  // create the right stages
  switch (currentStage) {

    case 0: {

      // go to the next stage
      currentStage++;

      // return the key stage
      return std::make_shared<MM3DShuffleStage>(n, num_nodes, num_threads);
    }
    case 1: {

      // go to the next stage
      currentStage++;

      // return the key stage
      return std::make_shared<MM3DMultiplyStage>(n, num_nodes, num_threads);
    }
  }

  throw runtime_error("Unrecognized stage. How did we get here?");
}

int32_t pdb::MM3D::numStages() const {
  return 2;
}

pdb::PDBPhysicalAlgorithmType pdb::MM3D::getAlgorithmType() {
  return MM3DType;
}

pdb::PDBCatalogSetContainerType pdb::MM3D::getOutputContainerType() {
  return PDB_CATALOG_SET_VECTOR_CONTAINER;
}
