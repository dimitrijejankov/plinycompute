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
  for(const auto &sourceNode : result.newSourceNodes) {
    sources.insert(std::make_pair(0, sourceNode));
    sourcesWithIDs[sourceNode->getSourcePageSet(sourcesWithIDs)->pageSetIdentifier] = std::make_pair(0, sourceNode);
  }

  // add the new page sets
  for(const auto &pageSet : result.newPageSets) {

    // insert the page set with the specified number of consumers
    activePageSets[pageSet.first] += pageSet.second;
  }

  // deallocate the old ones
  for(const auto &pageSet : result.consumedPageSets) {

    // if we have that page set (it is not a pdb set)
    if(activePageSets.find(pageSet) != activePageSets.end()) {

      // remove the page set
      activePageSets[pageSet]--;
    }
  }

  // add the page sets that are due to be removed
  for(auto it = activePageSets.begin(); it != activePageSets.end(); it++) {

    //  check if we should remove this one
    auto jt = it;
    if(jt->second == 0) {
      pageSetsToRemove.emplace_back(jt->first);
      activePageSets.erase(jt);
    }
  }

  // return the algorithm
  return result.runMe;
}

bool PDBPhysicalOptimizer::hasAlgorithmToRun() {
  return !sources.empty();
}

void PDBPhysicalOptimizer::updateStats() {

}

std::vector<PDBPageSetIdentifier> PDBPhysicalOptimizer::getPageSetsToRemove() {

  // empty out the page set
  auto tmp = std::move(pageSetsToRemove);
  pageSetsToRemove = std::vector<PDBPageSetIdentifier>();

  // return the value
  return std::move(tmp);
}

}

