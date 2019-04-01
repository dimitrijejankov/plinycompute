//
// Created by dimitrije on 3/20/19.
//

#ifndef PDB_SILLYAGG_H
#define PDB_SILLYAGG_H

#include <DepartmentTotal.h>
#include <Employee.h>
#include <AggregateComp.h>
#include "LambdaCreationFunctions.h"

namespace pdb {

// this points to the location of the hash table that stores the result of the aggregation
void *whereHashTableSits;

// aggregate relies on having two methods in the output type: getKey () and getValue ()
class SillyAgg : public AggregateComp<DepartmentTotal, Employee, String, double> {

 public:

  ENABLE_DEEP_COPY

  // the key type must have == and size_t hash () defined
  Lambda<String> getKeyProjection(Handle<Employee> aggMe) override {
    return makeLambdaFromMember (aggMe, department);
  }

  // the value type must have + defined
  Lambda<double> getValueProjection(Handle<Employee> aggMe) override {
    return makeLambdaFromMethod (aggMe, getSalary);
  }

};

}

#endif //PDB_SILLYAGG_H
