#include <PDBOptimizerSource.h>
#include <PDBAbstractPhysicalNode.h>
#include <physicalOptimizer/PDBOptimizerSource.h>

bool pdb::OptimizerSourceComparator::operator()(const pdb::OptimizerSource &lhs, const pdb::OptimizerSource &rhs) {

  // first compare them based on the size
  if(lhs.first != rhs.first) {
    return lhs.first < rhs.first;
  }

  // if the size is equal compare them on the
  return lhs.second->getNodeIdentifier() != rhs.second->getNodeIdentifier();
}

bool pdb::PageSetIdentifierComparator::operator()(const std::pair<size_t, std::string> &lhs, const std::pair<size_t, std::string> &rhs) {

  // first compare them based on the size
  if(lhs.first != rhs.first) {
    return lhs.first < rhs.first;
  }

  // if the size is equal compare them on the
  return lhs.second < rhs.second;
}
