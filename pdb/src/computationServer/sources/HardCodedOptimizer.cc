#include <HardCodedOptimizer.h>
#include <PDBJoin8Algorithm.h>

pdb::HardCodedOptimizer::HardCodedOptimizer(uint64_t computationID) : computationID(computationID) {}


pdb::Handle<pdb::PDBPhysicalAlgorithm> pdb::HardCodedOptimizer::getNextAlgorithm() {

  std::string in0 = "inputDataForSetScanner_0";
  std::string in1 = "inputDataForSetScanner_1";
  std::string in2 = "inputDataForSetScanner_2";
  std::string in3 = "inputDataForSetScanner_3";
  std::string in4 = "inputDataForSetScanner_4";
  std::string in5 = "inputDataForSetScanner_5";
  std::string in6 = "inputDataForSetScanner_6";
  std::string in7 = "inputDataForSetScanner_7";

  std::string out0 = "OutFor_0_methodCall_7JoinComp8_hashed";
  std::string out1 = "OutFor_0_self_9JoinComp8_hashed";
  std::string out2 = "OutFor_0_self_14JoinComp8_hashed";
  std::string out3 = "OutFor_0_methodCall_19JoinComp8_hashed";
  std::string out4 = "OutFor_0_self_24JoinComp8_hashed";
  std::string out5 = "OutFor_0_self_29JoinComp8_hashed";
  std::string out6 = "OutFor_0_self_34JoinComp8_hashed";
  std::string out7 = "OutFor_0_methodCall_39JoinComp8_hashed";

  PDBPrimarySource source;

  source.source = pdb::makeObject<PDBSourcePageSetSpec>();
  source.source->sourceType = PDBSourceType::SetScanSource;

  PDBSinkPageSetSpec sink;

  pdb::Handle<pdb::Vector<PDBSetObject>> setsToMaterialize = pdb::makeObject<pdb::Vector<PDBSetObject>>();

  return pdb::makeObject<PDBJoin8Algorithm>(std::make_pair<std::string, std::string>("myData", "A"),
                                            std::make_pair<std::string, std::string>("myData", "B"),
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
                                            out7);
}

bool pdb::HardCodedOptimizer::hasAlgorithmToRun() {
  static bool x = true;

  if(x == true) {
    x = false;
    return true;
  }

  return false;
}

vector<pdb::PDBPageSetIdentifier> pdb::HardCodedOptimizer::getPageSetsToRemove() {
  return vector<PDBPageSetIdentifier>();
}
