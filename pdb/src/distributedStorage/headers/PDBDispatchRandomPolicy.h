//
// Created by dimitrije on 2/7/19.
//

#ifndef PDB_PDBDISPATCHERRANDOMPOLICY_H
#define PDB_PDBDISPATCHERRANDOMPOLICY_H

#include <PDBDispatchPolicy.h>
#include <random>
#include <mutex>
#include <chrono>

namespace pdb {

class PDBDispatchRandomPolicy : public PDBDispatchPolicy {

public:

  PDBDispatchRandomPolicy() : generator((unsigned long) std::chrono::system_clock::now().time_since_epoch().count()) {}

  /**
   * Returns the next node we are about to send our stuff to
   * @param database - the name of the database the stuff needs to be sent to
   * @param set - the name of the set the stuff needs to be sent to
   * @param nodes - the active nodes we can send stuff to
   * @return - the node we decided to send it to
   */
  PDBCatalogNodePtr getNextNode(const std::string &database, const std::string &set, const std::vector<PDBCatalogNodePtr> &nodes) override;

private:

  // make the random number generator
  std::default_random_engine generator;

  // the mutex to sync this
  std::mutex m;
};

}



#endif //PDB_PDBDISPATCHERRANDOMPOLICY_H
