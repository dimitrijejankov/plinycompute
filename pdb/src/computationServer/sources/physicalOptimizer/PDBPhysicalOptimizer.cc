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

  // select a source and pop it
  auto source = *sources.begin();

  // runs the algorithm generation part
  auto result = source.second->generateAlgorithm(sourcesWithIDs);

  // remove the source we just used
  sources.erase(sources.begin());

  // go through each consumer of the output of this algorithm and add it to the sources
  for(const auto &sourceNode : result.second) {
    sources.insert(std::make_pair(0, sourceNode));
    sourcesWithIDs[sourceNode->getNodeIdentifier()] = std::make_pair(0, sourceNode);
  }

  // return the algorithm
  return result.first;
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

