#include <PDBJoin8Algorithm.h>
#include <physicalAlgorithms/PDBJoin8AlgorithmKeyStage.h>
#include <physicalAlgorithms/PDBJoin8AlgorithmState.h>
#include <ExJob.h>

pdb::PDBJoin8Algorithm::PDBJoin8Algorithm(const std::pair<std::string, std::string> &sourceSet,
                                          const std::pair<std::string, std::string> &sinkSet,
                                          const std::string &in0,
                                          const std::string &out0,
                                          const std::string &in1,
                                          const std::string &out1,
                                          const std::string &in2,
                                          const std::string &out2,
                                          const std::string &in3,
                                          const std::string &out3,
                                          const std::string &in4,
                                          const std::string &out4,
                                          const std::string &in5,
                                          const std::string &out5,
                                          const std::string &in6,
                                          const std::string &out6,
                                          const std::string &in7,
                                          const std::string &out7) : in0(in0),
                                                                     in1(in1),
                                                                     in2(in2),
                                                                     in3(in3),
                                                                     in4(in4),
                                                                     in5(in5),
                                                                     in6(in6),
                                                                     in7(in7),
                                                                     out0(out0),
                                                                     out1(out1),
                                                                     out2(out2),
                                                                     out3(out3),
                                                                     out4(out4),
                                                                     out5(out5),
                                                                     out6(out6),
                                                                     out7(out7) {

  source.database = sourceSet.first;
  source.set = sourceSet.second;

  std::cout << source.database << " " << source.set << '\n';
  sink.database = sinkSet.first;
  sink.set = sinkSet.second;

  setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::PDBJoin8Algorithm::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {

  //
  auto state = std::make_shared<PDBJoin8AlgorithmState>();

  // init the logger for this algorithm
  state->logger = make_shared<PDBLogger>("PDBJoin8Algorithm_" + std::to_string(job->computationID));

  return state;
}

vector<pdb::PDBPhysicalAlgorithmStagePtr> pdb::PDBJoin8Algorithm::getStages() const {

  return {std::make_shared<PDBJoin8AlgorithmKeyStage>(source,
                                                      in0,
                                                      out0,
                                                      in1,
                                                      out1,
                                                      in2,
                                                      out2,
                                                      in3,
                                                      out3,
                                                      in4,
                                                      out4,
                                                      in5,
                                                      out5,
                                                      in6,
                                                      out6,
                                                      in7,
                                                      out7)};
}

int32_t pdb::PDBJoin8Algorithm::numStages() const {
  return 1;
}

pdb::PDBPhysicalAlgorithmType pdb::PDBJoin8Algorithm::getAlgorithmType() {
  return pdb::PDBPhysicalAlgorithmType::EightWayJoin;
}

pdb::PDBCatalogSetContainerType pdb::PDBJoin8Algorithm::getOutputContainerType() {
  return pdb::PDBCatalogSetContainerType::PDB_CATALOG_SET_VECTOR_CONTAINER;
}

pdb::PDBJoin8Algorithm::PDBJoin8Algorithm() {}
