#include <HardCodedOptimizer.h>
#include <PDBJoin3Algorithm.h>

pdb::HardCodedOptimizer::HardCodedOptimizer(uint64_t computationID) : computationID(computationID) {}


pdb::Handle<pdb::PDBPhysicalAlgorithm> pdb::HardCodedOptimizer::getNextAlgorithm() {

  std::string in0 = "inputDataForSetScanner_0";
  std::string in1 = "inputDataForSetScanner_1";
  std::string in2 = "inputDataForSetScanner_2";

  // in0 hashed
  std::string out0 = "OutFor_0_self_1JoinComp2_hashed";

  // in1 hashed
  std::string out1 = "OutFor_0_self_3JoinComp2_hashed";

  // in2 hashed
  std::string out2 = "OutFor_0_self_3JoinComp4_hashed";

  // in1 and in2 hashed
  std::string out3 = "OutFor_0_self_1JoinComp4_hashed";

  // the final output
  std::string final = "aggOutForAggregationComp5_out";

  PDBPrimarySource source;

  source.source = pdb::makeObject<PDBSourcePageSetSpec>();
  source.source->sourceType = PDBSourceType::SetScanSource;

  PDBSinkPageSetSpec sink;

  pdb::Handle<pdb::Vector<PDBSetObject>> setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();

  return pdb::makeObject<PDBJoin3Algorithm>(std::make_pair<std::string, std::string>("myData", "A"),
                                            std::make_pair<std::string, std::string>("myData", "B"),
                                            std::make_pair<std::string, std::string>("myData", "C"),
                                            std::make_pair<std::string, std::string>("myData", "D"),
                                            in0,
                                            out0,
                                            in1,
                                            out1,
                                            in2,
                                            out2,
                                            out3,
                                            final);
}

bool pdb::HardCodedOptimizer::hasAlgorithmToRun() {
  static bool x = true;

  if(x) {
    x = false;
    return true;
  }

  return false;
}

vector<pdb::PDBPageSetIdentifier> pdb::HardCodedOptimizer::getPageSetsToRemove() {
  return vector<PDBPageSetIdentifier>();
}
