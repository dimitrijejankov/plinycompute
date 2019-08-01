//
// Created by dimitrije on 2/20/19.
//

#ifndef PDB_SCANSUPERVISORSET_H
#define PDB_SCANSUPERVISORSET_H

#include <Supervisor.h>
#include <SetScanner.h>
#include <LambdaCreationFunctions.h>
#include <VectorTupleSetIterator.h>

class ScanSupervisorSet : public pdb::SetScanner<pdb::Supervisor> {
public:

  ENABLE_DEEP_COPY

  ScanSupervisorSet(const std::string &db, const std::string &set) : SetScanner(db, set) {}

  ScanSupervisorSet() = default;
};

#endif //PDB_SCANSUPERVISORSET_H
