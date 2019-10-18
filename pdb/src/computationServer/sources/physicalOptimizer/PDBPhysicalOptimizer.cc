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

  do {

    // select a source and pop it
    auto source = *sources.begin();

    // runs the algorithm generation part
    auto result = source.second->generateAlgorithm(pageSetCosts);

    // remove the source we just used, and add it to the list of processed sources
    sources.erase(sources.begin());
    processedSources.push_back(source);

    // go through each consumer of the output of this algorithm and add it to the sources
    for(const auto &sourceNode : result.newSourceNodes) {

      // the iterator
      auto it = sources.insert(std::make_pair(std::make_shared<PDBCatalogSetStats>(), sourceNode));

      // store the iterator so we can update it.
      auto inputPageSets = sourceNode->getInputPageSets();
      for_each(inputPageSets.begin(), inputPageSets.end(), [&](const auto &pageSet) {
        sourcesUsingPageSet[pageSet.pageSetIdentifier] = it;
      });
    }

    // add the new page sets
    for(const auto &pageSet : result.newPageSets) {

      // insert the page set with the specified number of consumers
      activePageSets[pageSet.first] += pageSet.second;
      pageSetCosts[pageSet.first] = std::make_shared<PDBCatalogSetStats>(0, 0, 0, 0);
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
    for(auto it = activePageSets.begin(); it != activePageSets.end();) {

      //  check if we should remove this one
      auto jt = it++;
      if(jt->second == 0) {
        pageSetsToRemove.emplace_back(jt->first);
        activePageSets.erase(jt);
      }
    }

    // did we manage to generate the algorithm
    if(result.resultType == PDBPlanningResultType::GENERATED_ALGORITHM) {
      return result.runMe;
    }

  } while (hasAlgorithmToRun());

  // return the null
  return nullptr;
}

bool PDBPhysicalOptimizer::hasAlgorithmToRun() {
  return !sources.empty();
}

void PDBPhysicalOptimizer::updatePageSet(const PDBPageSetIdentifier &identifier, size_t size) {

  // update the cost in the page set costs
  pageSetCosts[identifier] = std::make_shared<PDBCatalogSetStats>(0, 0, size, 0);

  // check if we have a source
  auto it = sourcesUsingPageSet.find(identifier);
  if(it != sourcesUsingPageSet.end()) {

    // get the pointer to the optimizer node
    auto node = it->second->second;

    // get the stats
    auto stats = it->second->first;

    // replace the source with updated cost
    stats->setSize += size;
    sources.erase(it->second);
    sources.insert(std::make_pair(stats, node));
  }
}

std::vector<PDBPageSetIdentifier> PDBPhysicalOptimizer::getPageSetsToRemove() {

  // set the sizes of all the removed sets to 0
  for (const auto &i : pageSetsToRemove) {
    pageSetCosts[i] = std::make_shared<PDBCatalogSetStats>(0, 0, 0, 0);
  }

  // empty out the page set
  auto tmp = std::move(pageSetsToRemove);
  pageSetsToRemove = std::vector<PDBPageSetIdentifier>();

  // return the value
  return std::move(tmp);
}

}

