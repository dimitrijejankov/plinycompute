//
// Created by dimitrije on 3/20/19.
//

#ifndef PDB_SILLYQUERY_H
#define PDB_SILLYQUERY_H

#include <Employee.h>
#include <Supervisor.h>
#include <SelectionComp.h>
#include "LambdaCreationFunctions.h"

namespace pdb {

class SillyQuery : public SelectionComp<Employee, Supervisor> {

 public:

  ENABLE_DEEP_COPY

  Lambda<bool> getSelection(Handle<Supervisor> checkMe) override {
    return makeLambdaFromMethod (checkMe, getSteve) == makeLambdaFromMember (checkMe, me);
  }

  Lambda<Handle<Employee>> getProjection(Handle<Supervisor> checkMe) override{
    return makeLambdaFromMethod (checkMe, getMe);
  }

};

}

#endif //PDB_SILLYQUERY_H
