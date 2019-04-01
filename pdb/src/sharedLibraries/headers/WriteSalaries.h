#ifndef PDB_WRITESALARIES
#define PDB_WRITESALARIES

#include <Employee.h>
#include <SetWriter.h>
#include <LambdaCreationFunctions.h>
#include <VectorTupleSetIterator.h>
#include <VectorSink.h>

class WriteSalaries : public pdb::SetWriter<double> {

 public:

  ENABLE_DEEP_COPY

  WriteSalaries() = default;

  // below constructor is not required, but if we do not call setOutput() here, we must call
  // setOutput() later to set the output set
  WriteSalaries(std::string dbName, std::string setName) {
    this->setOutput(std::move(dbName), std::move(setName));
  }
};

#endif //PDB_WRITESALARIES
