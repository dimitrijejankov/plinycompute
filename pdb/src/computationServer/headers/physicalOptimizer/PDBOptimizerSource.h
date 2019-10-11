#pragma once

#include <memory>
#include <map>
#include <PDBCatalogSetStats.h>

namespace pdb {

class PDBAbstractPhysicalNode;
using PDBAbstractPhysicalNodePtr = std::shared_ptr<PDBAbstractPhysicalNode>;
using PDBAbstractPhysicalNodeWeakPtr = std::weak_ptr<PDBAbstractPhysicalNode>;

using PDBPageSetIdentifier = std::pair<size_t, std::string>;

using OptimizerSource = std::pair<pdb::PDBCatalogSetStatsPtr, PDBAbstractPhysicalNodePtr>;
class OptimizerSourceComparator
{
 public:
  bool operator() (const OptimizerSource &lhs, const OptimizerSource &rhs) const;
};

class PageSetIdentifierComparator
{
 public:
  bool operator() (const PDBPageSetIdentifier &lhs, const PDBPageSetIdentifier &rhs) const;
};

using PDBPageSetCosts = std::map<PDBPageSetIdentifier, pdb::PDBCatalogSetStatsPtr, PageSetIdentifierComparator>;

}