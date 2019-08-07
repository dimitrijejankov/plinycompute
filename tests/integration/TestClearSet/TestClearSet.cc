#include <PDBClient.h>
#include <SharedEmployee.h>
#include <GenericWork.h>
#include <ScanSupervisorSet.h>
#include <SillyQuery.h>
#include <SillyAgg.h>
#include <FinalQuery.h>
#include <WriteSalaries.h>
#include <gtest/gtest.h>
#include "ScanEmployeeSet.h"
#include "EmployeeBuiltInIdentitySelection.h"
#include "WriteBuiltinEmployeeSet.h"

using namespace pdb;

int main(int argc, char* argv[]) {

  const size_t blockSize = 64;

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libScanSupervisorSet.so");
  pdbClient.registerType("libraries/libSillyQuery.so");
  pdbClient.registerType("libraries/libSillyAgg.so");
  pdbClient.registerType("libraries/libFinalQuery.so");
  pdbClient.registerType("libraries/libWriteSalaries.so");

  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("chris_db");

  // now, create the input and output sets
  pdbClient.createSet<Supervisor>("chris_db", "chris_set");
  pdbClient.createSet<double>("chris_db", "output_set");

  /// 3. Fill in the data

  // the department
  std::string departmentPrefix(4, 'a');
  int numRecords = 0;
  int numSteve = 0;
  for(int j = 0; j < 5; j++) {

    // make the allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

    // write a bunch of supervisors to it
    pdb::Handle<pdb::Vector<pdb::Handle<pdb::Supervisor>>> supers = pdb::makeObject<pdb::Vector<pdb::Handle<pdb::Supervisor>>>();

    int i = 0;
    try {

      for (; true; i++) {

        Handle<Supervisor> super = makeObject<Supervisor>("Steve Stevens", numRecords, std::string(departmentPrefix) + std::to_string(numRecords), 1);
        numRecords++;

        supers->push_back(super);
        for (int k = 0; k < 10; k++) {

          Handle<Employee> temp;
          temp = makeObject<Employee>("Steve Stevens", numRecords, std::string(departmentPrefix) + std::to_string(numRecords), 1);

          (*supers)[i]->addEmp(temp);
        }
      }

    } catch (pdb::NotEnoughSpace &e) {


      // remove the last supervisor
      supers->pop_back();

      // increment steave
      numSteve += supers->size();

      // send the data twice so we aggregate two times each department
      pdbClient.sendData<Supervisor>("chris_db", "chris_set", supers);
      pdbClient.sendData<Supervisor>("chris_db", "chris_set", supers);
    }
  }

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  /// 5. create all of the computation objects and run the query

  // make the scan set
  Handle<Computation> myScanSet = makeObject<ScanSupervisorSet>("chris_db", "chris_set");

  // make the first filter
  Handle<Computation> myFilter = makeObject<SillyQuery>();
  myFilter->setInput(myScanSet);

  // make the aggregation
  Handle<Computation> myAgg = makeObject<SillyAgg>();
  myAgg->setInput(myFilter);

  // make the final filter
  Handle<Computation> myFinalFilter = makeObject<FinalQuery>();
  myFinalFilter->setInput(myAgg);

  // make the set writer
  Handle<Computation> myWrite = makeObject<WriteSalaries>("chris_db", "output_set");
  myWrite->setInput(myFinalFilter);

  /// 5. Run the query for the first time

  // run computations
  pdbClient.executeComputations({ myWrite });

  // grab the iterator
  auto it = pdbClient.getSetIterator<double>("chris_db", "output_set");

  int i = 0;
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();

    // should be 2 since we sent the same data twice
    if(*r != 2) {
      std::cout << "Record is not aggregated twice" << std::endl;
      break;
    }

    // go to the next one
    i++;
  }

  std::cout << "Got " << i << " records expected " << numSteve << std::endl;

  pdbClient.clearSet("chris_db", "output_set");

  // grab the iterator
  it = pdbClient.getSetIterator<double>("chris_db", "output_set");

  i = 0;
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();

    // should be 2 since we sent the same data twice
    if(*r != 2) {
      std::cout << "Record is not aggregated twice" << std::endl;
      break;
    }

    // go to the next one
    i++;
  }

  std::cout << "Got " << i << " records expected 0" << std::endl;

  // run computations
  pdbClient.executeComputations({ myWrite });

  /// 5. Get the set from the

  // grab the iterator
  it = pdbClient.getSetIterator<double>("chris_db", "output_set");

  i = 0;
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();

    // should be 2 since we sent the same data twice
    if(*r != 2) {
      std::cout << "Record is not aggregated twice" << std::endl;
      break;
    }

    // go to the next one
    i++;
  }

  std::cout << "Got " << i << " records expected " << numSteve << std::endl;

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}