//
// Created by dimitrije on 2/20/19.
//

#ifndef PDB_WRITEBUILTINEMPLOYEESET_H
#define PDB_WRITEBUILTINEMPLOYEESET_H

#include <Employee.h>
#include <SetWriter.h>
#include <LambdaCreationFunctions.h>
#include <VectorTupleSetIterator.h>
#include <VectorSink.h>

class WriteBuiltinEmployeeSet : public pdb::SetWriter<pdb::Employee> {

 public:

  ENABLE_DEEP_COPY

  WriteBuiltinEmployeeSet() = default;

  // below constructor is not required, but if we do not call setOutput() here, we must call
  // setOutput() later to set the output set
  WriteBuiltinEmployeeSet(std::string dbName, std::string setName) {
    this->setOutput(std::move(dbName), std::move(setName));
  }

  // eventually, this method should be moved into a class that works with the system to
  // iterate through pages that are pulled from disk/RAM by the system... a programmer
  // should not provide this particular method
  pdb::ComputeSinkPtr getComputeSink(TupleSpec &consumeMe, TupleSpec &projection, pdb::ComputePlan &plan) override {
    return std::make_shared<pdb::VectorSink<pdb::Employee>>(consumeMe, projection);
  }
};

#endif //PDB_WRITEBUILTINEMPLOYEESET_H
