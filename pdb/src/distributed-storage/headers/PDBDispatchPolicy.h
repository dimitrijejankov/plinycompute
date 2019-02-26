//
// Created by dimitrije on 2/7/19.
//

#ifndef PDB_DISPATCHERPOLICY_H
#define PDB_DISPATCHERPOLICY_H

#include <PDBCatalogNode.h>

namespace pdb {

class PDBDispatchPolicy;
using PDBDispatcherPolicyPtr = std::shared_ptr<PDBDispatchPolicy>;

class PDBDispatchPolicy {

public:

  /**
   * Returns the next node we are about to send our stuff to
   * @param database - the name of the database the stuff needs to be sent to
   * @param set - the name of the set the stuff needs to be sent to
   * @param nodes - the active nodes we can send stuff to
   * @return - the node we decided to send it to
   */
  virtual PDBCatalogNodePtr getNextNode(const std::string &database, const std::string &set, const std::vector<PDBCatalogNodePtr> &nodes) = 0;

};

}


#endif //PDB_DISPATCHERPOLICY_H
