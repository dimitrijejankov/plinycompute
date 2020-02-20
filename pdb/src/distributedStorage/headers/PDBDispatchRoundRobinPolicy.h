#pragma once

#include "PDBDispatchPolicy.h"
#include <mutex>

namespace pdb {

class PDBDispatchRoundRobinPolicy : public PDBDispatchPolicy {

 public:

  PDBDispatchRoundRobinPolicy() = default;

  /**
   * Returns the next node we are about to send our stuff to
   * @param database - the name of the database the stuff needs to be sent to
   * @param set - the name of the set the stuff needs to be sent to
   * @param nodes - the active nodes we can send stuff to
   * @return - the node we decided to send it to
   */
  PDBCatalogNodePtr getNextNode(const std::string &database, const std::string &set, const std::vector<PDBCatalogNodePtr> &nodes) override;

 private:

  // we use this to compare
  struct CompareSet {
    bool operator()(const std::pair<std::string, std::string>& a, const std::pair<std::string, std::string>& b) const {

      // check if equal
      if(a.first == b.first) {
        return a.second < b.second;
      }

      // check by db
      return a.first < b.first ;
    }
  };

  // this gives us the next node index
  std::map<std::pair<std::string, std::string>, uint32_t, CompareSet> nextRoundRobin;

  // the mutex to sync this
  std::mutex m;
};

}
