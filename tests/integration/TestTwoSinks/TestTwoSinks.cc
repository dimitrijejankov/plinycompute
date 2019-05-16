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
  pdbClient.createSet<double>("chris_db", "output_set1");
  pdbClient.createSet<double>("chris_db", "output_set2");

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
      pdbClient.sendData<Supervisor>("chris_db", "chris_set1", supers);
      pdbClient.sendData<Supervisor>("chris_db", "chris_set2", supers);
    }
  }

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  String myTCAPString =
      "inputData(in0) <= SCAN ('chris_db', 'chris_set', 'SetScanner_0')\n"
      "methodCall_0OutFor_SelectionComp1(in0,methodCall_0OutFor__getSteve) <= APPLY (inputData(in0), inputData(in0), 'SelectionComp_1', 'methodCall_0', [('inputTypeName', 'pdb::Supervisor'), ('lambdaType', 'methodCall'), ('methodName', 'getSteve'), ('returnTypeName', 'pdb::Supervisor')])\n"
      "attAccess_1OutForSelectionComp1(in0,methodCall_0OutFor__getSteve,att_1OutFor_me) <= APPLY (methodCall_0OutFor_SelectionComp1(in0), methodCall_0OutFor_SelectionComp1(in0,methodCall_0OutFor__getSteve), 'SelectionComp_1', 'attAccess_1', [('attName', 'me'), ('attTypeName', 'pdb::Handle&lt;pdb::Employee&gt;'), ('inputTypeName', 'pdb::Supervisor'), ('lambdaType', 'attAccess')])\n"
      "equals_2OutForSelectionComp1(in0,methodCall_0OutFor__getSteve,att_1OutFor_me,bool_2_1) <= APPLY (attAccess_1OutForSelectionComp1(methodCall_0OutFor__getSteve,att_1OutFor_me), attAccess_1OutForSelectionComp1(in0,methodCall_0OutFor__getSteve,att_1OutFor_me), 'SelectionComp_1', '==_2', [('lambdaType', '==')])\n"
      "filteredInputForSelectionComp1(in0) <= FILTER (equals_2OutForSelectionComp1(bool_2_1), equals_2OutForSelectionComp1(in0), 'SelectionComp_1')\n"
      "methodCall_3OutFor_SelectionComp1(in0,methodCall_3OutFor__getMe) <= APPLY (filteredInputForSelectionComp1(in0), filteredInputForSelectionComp1(in0), 'SelectionComp_1', 'methodCall_3', [('inputTypeName', 'pdb::Supervisor'), ('lambdaType', 'methodCall'), ('methodName', 'getMe'), ('returnTypeName', 'pdb::Supervisor')])\n"
      "deref_4OutForSelectionComp1 (methodCall_3OutFor__getMe) <= APPLY (methodCall_3OutFor_SelectionComp1(methodCall_3OutFor__getMe), methodCall_3OutFor_SelectionComp1(), 'SelectionComp_1', 'deref_4')\n"
      "attAccess_0OutForAggregationComp2(methodCall_3OutFor__getMe,att_0OutFor_department) <= APPLY (deref_4OutForSelectionComp1(methodCall_3OutFor__getMe), deref_4OutForSelectionComp1(methodCall_3OutFor__getMe), 'AggregationComp_2', 'attAccess_0', [('attName', 'department'), ('attTypeName', 'pdb::String'), ('inputTypeName', 'pdb::Employee'), ('lambdaType', 'attAccess')])\n"
      "deref_1OutForAggregationComp2(methodCall_3OutFor__getMe, att_0OutFor_department) <= APPLY (attAccess_0OutForAggregationComp2(att_0OutFor_department), attAccess_0OutForAggregationComp2(methodCall_3OutFor__getMe), 'AggregationComp_2', 'deref_1')\n"
      "aggWithValue(att_0OutFor_department,methodCall_2OutFor__getSalary) <= APPLY (deref_1OutForAggregationComp2(methodCall_3OutFor__getMe), deref_1OutForAggregationComp2(att_0OutFor_department), 'AggregationComp_2', 'methodCall_2', [('inputTypeName', 'pdb::Employee'), ('lambdaType', 'methodCall'), ('methodName', 'getSalary'), ('returnTypeName', 'pdb::Employee')])\n"
      "agg (aggOutFor2)<= AGGREGATE (aggWithValue(att_0OutFor_department, methodCall_2OutFor__getSalary),'AggregationComp_2')\n"
      "selectionOne(aggOutFor2,methodCall_0OutFor__checkSales) <= APPLY (agg(aggOutFor2), agg(aggOutFor2), 'SelectionComp_3', 'methodCall_0', [('inputTypeName', 'pdb::DepartmentTotal'), ('lambdaType', 'methodCall'), ('methodName', 'checkSales'), ('returnTypeName', 'pdb::DepartmentTotal')])\n"
      "selectionOneFilter(aggOutFor2) <= FILTER (selectionOne(methodCall_0OutFor__checkSales), selectionOne(aggOutFor2), 'SelectionComp_3')\n"
      "selectionOneFilterRemoved (methodCall_1OutFor__getTotSales) <= APPLY (selectionOneFilter(aggOutFor2), selectionOneFilter(), 'SelectionComp_3', 'methodCall_1', [('inputTypeName', 'pdb::DepartmentTotal'), ('lambdaType', 'methodCall'), ('methodName', 'getTotSales'), ('returnTypeName', 'pdb::DepartmentTotal')])\n"
      "selectionOneFilterRemoved_out( ) <= OUTPUT ( selectionOneFilterRemoved ( methodCall_1OutFor__getTotSales ), 'chris_db', 'chris_set2', 'SetWriter_4')\n"
      "selectionTwo(aggOutFor2,methodCall_0OutFor__checkSales) <= APPLY (agg(aggOutFor2), agg(aggOutFor2), 'SelectionComp_5', 'methodCall_0', [('inputTypeName', 'pdb::DepartmentTotal'), ('lambdaType', 'methodCall'), ('methodName', 'checkSales'), ('returnTypeName', 'pdb::DepartmentTotal')])\n"
      "selectionTwoFilter(aggOutFor2) <= FILTER (selectionTwo(methodCall_0OutFor__checkSales), selectionTwo(aggOutFor2), 'SelectionComp_5')\n"
      "selectionTwoFilterRemoved (methodCall_1OutFor__getTotSales) <= APPLY (selectionTwoFilter(aggOutFor2), selectionTwoFilter(), 'SelectionComp_5', 'methodCall_1', [('inputTypeName', 'pdb::DepartmentTotal'), ('lambdaType', 'methodCall'), ('methodName', 'getTotSales'), ('returnTypeName', 'pdb::DepartmentTotal')])\n"
      "selectionTwoFilterRemoved_out( ) <= OUTPUT ( selectionTwoFilterRemoved ( methodCall_1OutFor__getTotSales ), 'chris_db', 'chris_set1', 'SetWriter_6')\n";

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
  Handle<Computation> myFinalFilter1 = makeObject<FinalQuery>();
  myFinalFilter1->setInput(myAgg);

  // make the set writer
  Handle<Computation> myWrite1 = makeObject<WriteSalaries>();
  myWrite1->setInput(myFinalFilter1);

  // make the final filter
  Handle<Computation> myFinalFilter2 = makeObject<FinalQuery>();
  myFinalFilter2->setInput(myAgg);

  // make the set writer
  Handle<Computation> myWrite2 = makeObject<WriteSalaries>();
  myWrite2->setInput(myFinalFilter2);

  // put them in the list of computations
  myComputations->push_back(myScanSet);
  myComputations->push_back(myFilter);
  myComputations->push_back(myAgg);
  myComputations->push_back(myFinalFilter1);
  myComputations->push_back(myWrite1);
  myComputations->push_back(myFinalFilter2);
  myComputations->push_back(myWrite2);

  //TODO this is just a preliminary version of the execute computation before we add back the TCAP generation
  pdbClient.executeComputations(myComputations, myTCAPString);

  /// 5. Get the set from the

  // grab the iterator
  auto it = pdbClient.getSetIterator<double>("chris_db", "output_set1");

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

  std::cout << R"(Got for ("chris_db", "output_set1"))" << i << " records expected " << numSteve << std::endl;

  /// 6. Get the set from the

  // grab the iterator
  it = pdbClient.getSetIterator<double>("chris_db", "output_set2");

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

  std::cout << R"(Got for ("chris_db", "output_set2"))" << i << " records expected " << numSteve << std::endl;

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}