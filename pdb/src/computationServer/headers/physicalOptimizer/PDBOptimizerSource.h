#pragma once

#include <memory>
#include <map>

namespace pdb {

class PDBAbstractPhysicalNode;
using PDBAbstractPhysicalNodePtr = std::shared_ptr<PDBAbstractPhysicalNode>;
using PDBAbstractPhysicalNodeWeakPtr = std::weak_ptr<PDBAbstractPhysicalNode>;

using OptimizerSource = std::pair<size_t, PDBAbstractPhysicalNodePtr>;
class OptimizerSourceComparator
{
 public:
  bool operator() (const OptimizerSource &lhs, const OptimizerSource &rhs);
};

class PageSetIdentifierComparator
{
 public:
  bool operator() (const std::pair<size_t, std::string> &lhs, const std::pair<size_t, std::string> &rhs);
};

using sourceCosts = std::map<std::pair<size_t, std::string>, OptimizerSource, PageSetIdentifierComparator>;

}