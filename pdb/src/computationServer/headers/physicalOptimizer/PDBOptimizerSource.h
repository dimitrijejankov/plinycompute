#pragma once

#include <memory>
#include <map>

namespace pdb {

class PDBAbstractPhysicalNode;
using PDBAbstractPhysicalNodePtr = std::shared_ptr<PDBAbstractPhysicalNode>;
using PDBAbstractPhysicalNodeWeakPtr = std::weak_ptr<PDBAbstractPhysicalNode>;

using PDBPageSetIdentifier = std::pair<size_t, std::string>;

using OptimizerSource = std::pair<size_t, PDBAbstractPhysicalNodePtr>;
class OptimizerSourceComparator
{
 public:
  bool operator() (const OptimizerSource &lhs, const OptimizerSource &rhs);
};

class PageSetIdentifierComparator
{
 public:
  bool operator() (const PDBPageSetIdentifier &lhs, const PDBPageSetIdentifier &rhs);
};

using PDBPageSetCosts = std::map<PDBPageSetIdentifier, size_t, PageSetIdentifierComparator>;

}