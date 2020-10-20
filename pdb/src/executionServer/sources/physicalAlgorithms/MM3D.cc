#include <MM3D.h>

pdb::MM3D::MM3D(int32_t n, int32_t num_nodes, int32_t num_threads) : n(n),
                                                                     num_nodes(num_nodes),
                                                                     num_threads(num_threads) {}

pdb::PDBPhysicalAlgorithmStatePtr pdb::MM3D::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  return PDBPhysicalAlgorithm::getInitialState(job);
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::MM3D::getNextStage(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
  return PDBPhysicalAlgorithm::getNextStage(state);
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
