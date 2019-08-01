//
// Created by dimitrije on 3/20/19.
//

#ifndef PDB_FINALQUERY_H
#define PDB_FINALQUERY_H

#include <SelectionComp.h>
#include <DepartmentTotal.h>
#include "LambdaCreationFunctions.h"

namespace pdb {

class FinalQuery : public SelectionComp<double, DepartmentTotal> {

public:

  ENABLE_DEEP_COPY

 public:

  Lambda<bool> getSelection(Handle<DepartmentTotal> checkMe) override {
    return makeLambdaFromMethod (checkMe, checkSales);
  }

  Lambda<Handle<double>> getProjection(Handle<DepartmentTotal> checkMe) override {
    return makeLambdaFromMethod (checkMe, getTotSales);
  }

};

}

#endif //PDB_FINALQUERY_H
