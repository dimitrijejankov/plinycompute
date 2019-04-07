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

  String myTCAPString =
  "inputData (in) <= SCAN ('chris_db', 'chris_set', 'SetScanner_0', []) \n"
  "inputWithAtt (in, att) <= APPLY (inputData (in), inputData (in), 'SelectionComp_1', 'methodCall_0', []) \n"
  "inputWithAttAndMethod (in, att, method) <= APPLY (inputWithAtt (in), inputWithAtt (in, att), 'SelectionComp_1', 'attAccess_1', []) \n"
  "inputWithBool (in, bool) <= APPLY (inputWithAttAndMethod (att, method), inputWithAttAndMethod (in), 'SelectionComp_1', '==_2', []) \n"
  "filteredInput (in) <= FILTER (inputWithBool (bool), inputWithBool (in), 'SelectionComp_1', []) \n"
  "projectedInputWithPtr (out) <= APPLY (filteredInput (in), filteredInput (), 'SelectionComp_1', 'methodCall_3', []) \n"
  "projectedInput (out) <= APPLY (projectedInputWithPtr (out), projectedInputWithPtr (), 'SelectionComp_1', 'deref_4', []) \n"
  "aggWithKeyWithPtr (out, key) <= APPLY (projectedInput (out), projectedInput (out), 'AggregationComp_2', 'attAccess_0', []) \n"
  "aggWithKey (out, key) <= APPLY (aggWithKeyWithPtr (key), aggWithKeyWithPtr (out), 'AggregationComp_2', 'deref_1', []) \n"
  "aggWithValue (key, value) <= APPLY (aggWithKey (out), aggWithKey (key), 'AggregationComp_2', 'methodCall_2', []) \n"
  "agg (aggOut) <=	AGGREGATE (aggWithValue (key, value), 'AggregationComp_2', []) \n"
  "checkSales (aggOut, isSales) <= APPLY (agg (aggOut), agg (aggOut), 'SelectionComp_3', 'methodCall_0', []) \n"
  "justSales (aggOut, isSales) <= FILTER (checkSales (isSales), checkSales (aggOut), 'SelectionComp_3', []) \n"
  "final (result) <= APPLY (justSales (aggOut), justSales (), 'SelectionComp_3', 'methodCall_1', []) \n"
  "write () <= OUTPUT (final (result), 'chris_db', 'output_set', 'SetWriter_4', [])";

  // here is the list of computations
  Handle<Vector<Handle<Computation>>> myComputations = makeObject<Vector<Handle<Computation>>>();

  /// 5. create all of the computation objects and run the query

  // make the scan set
  Handle<Computation> myScanSet = makeObject<ScanSupervisorSet>();

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
  Handle<Computation> myWrite = makeObject<WriteSalaries>();
  myWrite->setInput(myFinalFilter);

  // put them in the list of computations
  myComputations->push_back(myScanSet);
  myComputations->push_back(myFilter);
  myComputations->push_back(myAgg);
  myComputations->push_back(myFinalFilter);
  myComputations->push_back(myWrite);

  //TODO this is just a preliminary version of the execute computation before we add back the TCAP generation
  pdbClient.executeComputations(myComputations, myTCAPString);

  /// 5. Get the set from the

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

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}