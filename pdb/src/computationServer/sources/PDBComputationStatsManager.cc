//
// Created by dimitrije on 2/28/19.
//

#include <PDBComputationStatsManager.h>

uint64_t PDBComputationStatsManager::startComputation() {

  // lock the stuff
  std::unique_lock<std::mutex> lock(computationIDLock);

  // grab a new computation id
  auto compID = lastComputationID++;

  // init the stat structure
  auto stat = std::make_shared<PDBComputationStats>();
  stat->id = compID;
  stat->stillRunning = true;
  stat->start = clock();

  // store the stat
  stats.insert(std::make_pair(compID, stat));

  return 0;
}

void PDBComputationStatsManager::endComputation(uint64_t compID) {

  // lock the stuff
  std::unique_lock<std::mutex> lock(computationIDLock);

  // grab an iterator to the struct thingy
  auto it = stats.find(compID);

  // mark it as finished
  it->second->stillRunning = false;
  it->second->end = clock();
}
