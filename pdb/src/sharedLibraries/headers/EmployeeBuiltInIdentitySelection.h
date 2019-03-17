//
// Created by dimitrije on 2/20/19.
//

#ifndef PDB_EMPLOYEEBUILTINIDENTITYSELECTION_H
#define PDB_EMPLOYEEBUILTINIDENTITYSELECTION_H

#include <SelectionComp.h>
#include <Employee.h>
#include "LambdaCreationFunctions.h"

class EmployeeBuiltInIdentitySelection : public pdb::SelectionComp<pdb::Employee, pdb::Employee> {

 public:

  ENABLE_DEEP_COPY

  EmployeeBuiltInIdentitySelection() = default;

   pdb::Lambda<bool> getSelection(pdb::Handle<pdb::Employee> &checkMe) override {
    return makeLambda(checkMe, [](pdb::Handle<pdb::Employee>& checkMe) { return true; });
  }

  // The Lambda defined here will return a new Employee with the same name as the old one, and age 100.
  pdb::Lambda<pdb::Handle<pdb::Employee>> getProjection(pdb::Handle<pdb::Employee> &checkMe) override {
    return makeLambda(checkMe, [](pdb::Handle<pdb::Employee>& checkMe) {
      pdb::Handle<pdb::Employee> newEmployee = pdb::makeObject<pdb::Employee>(*(checkMe->getName()), 100);  // cannot get age!
      return newEmployee;
    });
  }

};

#endif //PDB_EMPLOYEEBUILTINIDENTITYSELECTION_H
