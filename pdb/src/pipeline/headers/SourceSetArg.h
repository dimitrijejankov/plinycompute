#include <utility>

#pragma once

#include <ComputeInfo.h>
#include <PDBCatalogSet.h>

namespace pdb {

class SourceSetArg;
using SourceSetArgPtr = std::shared_ptr<SourceSetArg>;

class SourceSetArg : public ComputeInfo {
 public:

  explicit SourceSetArg(PDBCatalogSetPtr set) : set(std::move(set)) {}

  /**
   * The pdb set we are passing as an argument
   */
  PDBCatalogSetPtr set;
};

}
