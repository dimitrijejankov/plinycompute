//
// Created by dimitrije on 2/28/19.
//

#ifndef PDB_PDBCOMPUTATIONSTATS_H
#define PDB_PDBCOMPUTATIONSTATS_H

#include <mutex>
#include <unordered_map>
#include <memory>

// just declaring a pointer type for the PDBComputationStats
struct PDBComputationStats;
using PDBComputationStatsPtr = std::shared_ptr<PDBComputationStats>;

/**
 * This is used to store all the stats about a computation
 */
struct PDBComputationStats {

  /**
   * The id of the computation
   */
  int64_t id = -1;

  /**
   * When was this computation started
   */
  clock_t start = 0;

  /**
   * When did it end
   */
  clock_t end = 0;

  /**
   * Is the computation still running
   */
  bool stillRunning = false;
};

/**
 * This manages the stats about each computation run by the computation server.
 * It is thread safe to use this class
 *
 * //TODO this should preserve state after reset...
 */
class PDBComputationStatsManager {


public:

  /**
   * Has to be called to start tracking of a computation. Basically it assigns an id to a computation
   * that is being executed
   * @return the id that is assigned
   */
  uint64_t startComputation();

  /**
   * This is called to end the computation with a particular id
   */
  void endComputation(uint64_t compID);

private:

  /**
   * Computation id associated with the stats about that computation
   */
  std::unordered_map<uint64_t, PDBComputationStatsPtr> stats;

  /**
   * The last job id we run
   */
  uint64_t lastComputationID = 0;

  /**
   * Used to lock the data structures that keep
   */
  std::mutex computationIDLock;

};

#endif //PDB_PDBCOMPUTATIONSTATS_H
