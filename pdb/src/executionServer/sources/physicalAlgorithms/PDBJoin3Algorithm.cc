#include <PDBJoin3Algorithm.h>
#include <physicalAlgorithms/PDBJoin3AlgorithmKeyStage.h>
#include <physicalAlgorithms/PDBJoin3AggAsyncStage.h>
#include <physicalAlgorithms/PDBJoin3AlgorithmState.h>
#include <physicalAlgorithms/PDBJoin3AlgorithmJoinStage.h>
#include <ExJob.h>
#include <PDBJoin3AlgorithmState.h>

pdb::PDBJoin3Algorithm::PDBJoin3Algorithm(const std::pair<std::string, std::string> &sourceSet0,
                                          const std::pair<std::string, std::string> &sourceSet1,
                                          const std::pair<std::string, std::string> &sourceSet2,
                                          const std::pair<std::string, std::string> &sinkSet,
                                          const std::string &in0,
                                          const std::string &out0,
                                          const std::string &in1,
                                          const std::string &out1,
                                          const std::string &in2,
                                          const std::string &out2,
                                          const std::string &out3,
                                          const std::string &final) : in0(in0),
                                                                      in1(in1),
                                                                      in2(in2),
                                                                      out0(out0),
                                                                      out1(out1),
                                                                      out2(out2),
                                                                      out3(out3),
                                                                      final(final) {

  source0.database = sourceSet0.first;
  source0.set = sourceSet0.second;
  std::cout << "source0" << source0.database << " " << source0.set << '\n';

  source1.database = sourceSet1.first;
  source1.set = sourceSet1.second;
  std::cout << "source1" << source1.database << " " << source1.set << '\n';

  source2.database = sourceSet2.first;
  source2.set = sourceSet2.second;
  std::cout << "source2" << source2.database << " " << source2.set << '\n';

  sink.database = sinkSet.first;
  sink.set = sinkSet.second;
  std::cout << "sink" << sink.database << " " << sink.set << '\n';

  setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();
}

pdb::PDBPhysicalAlgorithmStatePtr pdb::PDBJoin3Algorithm::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {

  //
  auto state = std::make_shared<PDBJoin3AlgorithmState>();

  // init the logger for this algorithm
  state->logger = make_shared<PDBLogger>("PDBJoin8Algorithm_" + std::to_string(job->computationID));

  return state;
}

pdb::PDBPhysicalAlgorithmStagePtr pdb::PDBJoin3Algorithm::getNextStage(const PDBPhysicalAlgorithmStatePtr &state) {

  if(curStage == 0) {

    curStage++;
    return std::make_shared<PDBJoin3AlgorithmKeyStage>(source0,
                                                       source1,
                                                       source2,
                                                       in0,
                                                       out0,
                                                       in1,
                                                       out1,
                                                       in2,
                                                       out2,
                                                       out3,
                                                       final);
  }
  else if(curStage == 1) {

    curStage++;
    return std::make_shared<PDBJoin3AggAsyncStage>(source0,
                                                   source1,
                                                   source2,
                                                   in0,
                                                   out0,
                                                   in1,
                                                   out1,
                                                   in2,
                                                   out2,
                                                   out3,
                                                   final);
  }

  return nullptr;
}

int32_t pdb::PDBJoin3Algorithm::numStages() const {
  return 2;
}

pdb::PDBPhysicalAlgorithmType pdb::PDBJoin3Algorithm::getAlgorithmType() {
  return pdb::PDBPhysicalAlgorithmType::EightWayJoin;
}

pdb::PDBCatalogSetContainerType pdb::PDBJoin3Algorithm::getOutputContainerType() {
  return pdb::PDBCatalogSetContainerType::PDB_CATALOG_SET_VECTOR_CONTAINER;
}

pdb::PDBJoin3Algorithm::PDBJoin3Algorithm() = default;
