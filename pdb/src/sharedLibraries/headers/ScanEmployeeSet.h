//
// Created by dimitrije on 2/20/19.
//

#ifndef PDB_SCANEMPLOYEESET_H
#define PDB_SCANEMPLOYEESET_H

#include <Employee.h>
#include <SetScanner.h>
#include <LambdaCreationFunctions.h>
#include <VectorTupleSetIterator.h>

class ScanEmployeeSet : public pdb::SetScanner<pdb::Employee> {

 public:

  ENABLE_DEEP_COPY

  ScanEmployeeSet() = default;

  ScanEmployeeSet(std::string& dbname, std::string& setname) : pdb::SetScanner<pdb::Employee>(dbname, setname) {}
};

#endif //PDB_SCANEMPLOYEESET_H
