//
// Created by vicram on 6/25/19.
//

#ifndef PDB_WRITEDEPARTMENTMAXSET_H
#define PDB_WRITEDEPARTMENTMAXSET_H

#include <DepartmentMax.h>
#include <SetWriter.h>
namespace pdb{
class WriteDepartmentMaxSet : public SetWriter<DepartmentMax> {
 public:
  ENABLE_DEEP_COPY

  WriteDepartmentMaxSet() = default;

  // below constructor is not required, but if we do not call setOutputSet() here, we must call
  // setOutputSet() later to set the output set
  WriteDepartmentMaxSet(std::string dbName, std::string setName) {
    this->setOutputSet(std::move(dbName), std::move(setName));
  }
};
}


#endif //PDB_WRITEDEPARTMENTMAXSET_H
