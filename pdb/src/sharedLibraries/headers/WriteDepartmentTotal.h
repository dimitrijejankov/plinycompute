//
// Created by vicram on 6/3/19.
//

#ifndef PDB_WRITEDEPARTMENTTOTAL_H
#define PDB_WRITEDEPARTMENTTOTAL_H

#include <SetWriter.h>
#include <DepartmentTotal.h>

namespace pdb {

class WriteDepartmentTotal : public SetWriter<DepartmentTotal> {
 public:
  ENABLE_DEEP_COPY
  WriteDepartmentTotal() = default;
  WriteDepartmentTotal(const String &dbName, const String &setName) : SetWriter(dbName, setName) {}
};

}

#endif //PDB_WRITEDEPARTMENTTOTAL_H
