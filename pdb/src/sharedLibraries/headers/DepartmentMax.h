//
// Created by vicram on 6/25/19.
//

#ifndef PDB_DEPARTMENTMAX_H
#define PDB_DEPARTMENTMAX_H

#include "Object.h"
#include "PDBString.h"

namespace pdb {
class DepartmentMax : public Object {
 public:
  int max;
  String departmentName;

  ENABLE_DEEP_COPY

  String& getKey() {
    return departmentName;
  }

  int& getValue() {
    return max;
  }
};
}

#endif //PDB_DEPARTMENTMAX_H
