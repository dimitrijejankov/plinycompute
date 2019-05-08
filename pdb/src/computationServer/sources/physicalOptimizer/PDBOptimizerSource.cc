#include <PDBOptimizerSource.h>
#include <PDBAbstractPhysicalNode.h>

bool pdb::OptimizerSourceComparator::operator()(const pdb::OptimizerSource &lhs, const pdb::OptimizerSource &rhs) {

  // first compare them based on the size
  if(lhs.first != rhs.first) {
    return lhs.first < rhs.first;
  }

  // if the size is equal compare them on the
  return lhs.second->getNodeIdentifier() != rhs.second->getNodeIdentifier();
}