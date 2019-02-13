//
// Created by dimitrije on 10/16/18.
//

#ifndef PDB_STOPINPAGERESULT_H
#define PDB_STOPINPAGERESULT_H

// PRELOAD %BufPinPageResult%

#include "PDBString.h"
#include "PDBSet.h"

namespace pdb {

// request to get an anonymous page
class BufPinPageResult : public Object {

public:

  BufPinPageResult(const size_t &offset, const bool success) : offset(offset), success(success) {}

  BufPinPageResult() = default;

  ~BufPinPageResult() = default;

  ENABLE_DEEP_COPY;

  // a pointer to the raw bytes
  uint64_t offset = 0;

  // did we succeed
  bool success = false;
};
}

#endif //PDB_STOPINPAGEREQUEST_H
