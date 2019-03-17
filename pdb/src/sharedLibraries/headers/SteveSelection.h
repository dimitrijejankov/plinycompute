//
// Created by vicram on 3/17/19.
//

#ifndef PDB_STEVESELECTION_H
#define PDB_STEVESELECTION_H

#include <SelectionComp.h>
#include <Employee.h>
#include <Supervisor.h>
#include "LambdaCreationFunctions.h"

 class SteveSelection : public pdb::SelectionComp<pdb::Employee, pdb::Supervisor> {
 public:

  ENABLE_DEEP_COPY

  // This predicate is true iff the Supervisor's name is "Steve Stevens" AND the Supervisor
  // contains an Employee named "Steve Stevens" in his myGuys vector.
  pdb::Lambda<bool> getSelection(pdb::Handle<pdb::Supervisor> &checkMe) override {
    return makeLambdaFromMethod (checkMe, getSteve) == makeLambdaFromMember (checkMe, me);
  }

  pdb::Lambda<pdb::Handle<pdb::Employee>> getProjection(pdb::Handle<pdb::Supervisor> &checkMe) override {
    return makeLambdaFromMethod (checkMe, getMe);
  }
};

#endif //PDB_STEVESELECTION_H
