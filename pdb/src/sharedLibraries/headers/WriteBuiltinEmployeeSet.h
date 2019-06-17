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

  // below constructor is not required, but if we do not call setOutputSet() here, we must call
  // setOutputSet() later to set the output set
  WriteBuiltinEmployeeSet(std::string dbName, std::string setName) {
    this->setOutputSet(std::move(dbName), std::move(setName));
  }
};

#endif //PDB_WRITEBUILTINEMPLOYEESET_H
