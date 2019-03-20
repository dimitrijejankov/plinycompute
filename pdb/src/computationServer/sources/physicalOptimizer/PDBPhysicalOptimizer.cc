//
// Created by dimitrije on 2/21/19.
//

#include <physicalOptimizer/PDBPhysicalOptimizer.h>
#include <AtomicComputationList.h>
#include <Lexer.h>
#include <Parser.h>
#include <SetScanner.h>
#include <AtomicComputationClasses.h>
#include <PDBCatalogClient.h>

namespace pdb {

pdb::Handle<pdb::PDBPhysicalAlgorithm> PDBPhysicalOptimizer::getNextAlgorithm() {

  // select a source
  auto source = sources.top();
  sources.pop();

  // runs the algorithm generation part
  return source.second->generateAlgorithm();
}

bool PDBPhysicalOptimizer::hasAlgorithmToRun() {
  return !sources.empty();
}

void PDBPhysicalOptimizer::updateStats() {

}

std::vector<pair<uint64_t, std::string>> PDBPhysicalOptimizer::getPageSetsToRemove() {

  /// TODO this is just here to test the
  vector<pair<uint64_t, string>> ret;
  ret.reserve(2);

  // just add some page set identifiers for the selection
  ret.emplace_back(std::make_pair(computationID, "inputDataForScanSet_0"));
  ret.emplace_back(std::make_pair(computationID, "nativ_1OutForSelectionComp1_out"));

  return std::move(ret);
}


}

