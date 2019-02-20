/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#include <utility>
#include "Handle.h"
#include "Lambda.h"
#include "Supervisor.h"
#include "Employee.h"
#include "LambdaCreationFunctions.h"
#include "UseTemporaryAllocationBlock.h"
#include "Pipeline.h"
#include "SetWriter.h"
#include "SelectionComp.h"
#include "AggregateComp.h"
#include "ScanSet.h"
#include "DepartmentTotal.h"
#include "VectorSink.h"
#include "HashSink.h"
#include "MapTupleSetIterator.h"
#include "VectorTupleSetIterator.h"
#include "ComputePlan.h"

// to run the aggregate, the system first passes each through the hash operation...
// then the system
using namespace pdb;

class WriteBuiltinEmployeeSet : public SetWriter<Employee> {

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
  ComputeSinkPtr getComputeSink(TupleSpec &consumeMe, TupleSpec &projection, ComputePlan &plan) override {
    return std::make_shared<VectorSink<double>>(consumeMe, projection);
  }
};

class EmployeeBuiltInIdentitySelection : public SelectionComp<Employee, Employee> {

 public:

  ENABLE_DEEP_COPY

  EmployeeBuiltInIdentitySelection() = default;

  Lambda<bool> getSelection(Handle<Employee> &checkMe) override {
    return makeLambda(checkMe, [](Handle<Employee>& checkMe) { return true; });
  }

  Lambda<Handle<Employee>> getProjection(Handle<Employee> &checkMe) override {
    return makeLambda(checkMe, [](Handle<Employee>& checkMe) {
      Handle<Employee> newEmployee = makeObject<Employee>(*(checkMe->getName()), 100);  // cannot get age!
      return newEmployee;
    });
  }

};

class ScanEmployeeSet : public ScanSet<Employee> {

 public:

  ENABLE_DEEP_COPY

  ScanEmployeeSet() = default;

  // eventually, this method should be moved into a class that works with the system to
  // iterate through pages that are pulled from disk/RAM by the system... a programmer
  // should not provide this particular method
  ComputeSourcePtr getComputeSource(TupleSpec &schema, ComputePlan &plan) override {

    return std::make_shared<VectorTupleSetIterator>(

        // constructs a list of data objects to iterate through
        []() -> void * {

          // this implementation only serves six pages
          static int numPages = 0;
          if (numPages == 6)
            return nullptr;

          // create a page, loading it with random data
          void *myPage = malloc(1024 * 1024);
          {
            const UseTemporaryAllocationBlock tempBlock{myPage, 1024 * 1024};

            // write a bunch of supervisors to it
            Handle<Vector<Handle<Employee>>> employees = makeObject<Vector<Handle<Employee>>>();

            // this will build up the department
            char first = 'A', second = 'B';
            char myString[3];
            myString[2] = 0;

            try {
              for (int i = 0; true; i++) {

                myString[0] = first;
                myString[1] = second;

                // this will allow us to cycle through "AA", "AB", "AC", "BA", ...
                first++;
                if (first == 'D') {
                  first = 'A';
                  second++;
                  if (second == 'D')
                    second = 'A';
                }

                if(i % 2 == 0) {

                  Handle<Employee> temp = makeObject<Employee>("Steve Stevens", 20 + ((i) % 29), std::string(myString), i * 3.54);
                  employees->push_back(temp);
                }
                else {
                  Handle<Employee> temp = makeObject<Employee>("Ninja Turtles", 20 + ((i) % 29), std::string(myString), i * 3.54);
                  employees->push_back(temp);
                }
              }
            } catch (NotEnoughSpace &e) {

              getRecord (employees);
            }
          }
          numPages++;
          return myPage;
        },

        // frees the list of data objects that have been iterated
        [](void *freeMe) -> void {
          free(freeMe);
        },

        // and this is the chunk size, or number of items to put into each tuple set
        24);
  }
};

int main() {

  // this is the object allocation block where all of this stuff will reside
  makeObjectAllocatorBlock(1024 * 1024, true);

  // here is the list of computations
  Vector<Handle<Computation>> myComputations;

  Handle<Computation> myScanSet = makeObject<ScanEmployeeSet>();
  Handle<Computation> myQuery = makeObject<EmployeeBuiltInIdentitySelection>();
  myQuery->setInput(myScanSet);
  Handle<Computation> myWriteSet = makeObject<WriteBuiltinEmployeeSet>("by8_db", "output_set");
  myWriteSet->setInput(myQuery);

  // put them in the list of computations
  myComputations.push_back(myScanSet);
  myComputations.push_back(myQuery);
  myComputations.push_back(myWriteSet);

  // now we create the TCAP string
  String myTCAPString =
      "inputDataForScanSet_0(in0) <= SCAN ('input_set', 'by8_db', 'ScanSet_0') \n"\
      "nativ_0OutForSelectionComp1(in0,nativ_0_1OutFor) <= APPLY (inputDataForScanSet_0(in0), inputDataForScanSet_0(in0), 'SelectionComp_1', 'native_lambda_0', [('lambdaType', 'native_lambda')]) \n"\
      "filteredInputForSelectionComp1(in0) <= FILTER (nativ_0OutForSelectionComp1(nativ_0_1OutFor), nativ_0OutForSelectionComp1(in0), 'SelectionComp_1') \n"\
      "nativ_1OutForSelectionComp1 (nativ_1_1OutFor) <= APPLY (filteredInputForSelectionComp1(in0), filteredInputForSelectionComp1(), 'SelectionComp_1', 'native_lambda_1', [('lambdaType', 'native_lambda')]) \n"\
      "nativ_1OutForSelectionComp1_out( ) <= OUTPUT ( nativ_1OutForSelectionComp1 ( nativ_1_1OutFor ), 'output_set', 'by8_db', 'SetWriter_2') \n"\
      "inputDataForScanSet_0(in0) <= SCAN ('input_set', 'by8_db', 'ScanSet_0') \n"\
      "nativ_0OutForSelectionComp1(in0,nativ_0_1OutFor) <= APPLY (inputDataForScanSet_0(in0), inputDataForScanSet_0(in0), 'SelectionComp_1', 'native_lambda_0', [('lambdaType', 'native_lambda')]) \n"\
      "filteredInputForSelectionComp1(in0) <= FILTER (nativ_0OutForSelectionComp1(nativ_0_1OutFor), nativ_0OutForSelectionComp1(in0), 'SelectionComp_1') \n"\
      "nativ_1OutForSelectionComp1 (nativ_1_1OutFor) <= APPLY (filteredInputForSelectionComp1(in0), filteredInputForSelectionComp1(), 'SelectionComp_1', 'native_lambda_1', [('lambdaType', 'native_lambda')]) \n"\
      "nativ_1OutForSelectionComp1_out( ) <= OUTPUT ( nativ_1OutForSelectionComp1 ( nativ_1_1OutFor ), 'output_set', 'by8_db', 'SetWriter_2') \n";

  // and create a query object that contains all of this stuff
  Handle<ComputePlan> myPlan = makeObject<ComputePlan>(myTCAPString, myComputations);
  LogicalPlanPtr logicalPlan = myPlan->getPlan();
  AtomicComputationList computationList = logicalPlan->getComputations();
  std::cout << "to print logical plan:" << std::endl;
  std::cout << computationList << std::endl;

  // now, let's pretend that myPlan has been sent over the network, and we want to execute it... first we build
  // a pipeline into the aggregation operation
  PipelinePtr myPipeline = myPlan->buildPipeline(
      std::string("inputDataForScanSet_0"), /* this is the TupleSet the pipeline starts with */
      std::string("nativ_1OutForSelectionComp1"),     /* this is the TupleSet the pipeline ends with */
      std::string("SetWriter_2"), /* and since multiple Computation objects can consume the */
      /* same tuple set, we apply the Computation as well */

      // this lambda supplies new temporary pages to the pipeline
      []() -> std::pair<void *, size_t> {
        void *myPage = malloc(64 * 1024);
        return std::make_pair(myPage, 64 * 1024);
      },

      // this lambda frees temporary pages that do not contain any important data
      [](void *page) {
        free(page);
      },

      // and this lambda remembers the page that *does* contain important data...
      // in this simple aggregation, that one page will contain the hash table with
      // all of the aggregated data.
      [](void *page) {
        std::cout << "\nAsked to save page at address " << (size_t) page << "!!!\n";
        std::cout << "This should have a bunch of employees on it... let's see.\n";
        Handle<Vector<Handle<Employee>>> myHashTable = ((Record<Vector<Handle<Employee>>> *) page)->getRootObject();
        for (int i = 0; i < myHashTable->size(); i++) {
          std::cout << "Got employee " << *(((*myHashTable)[i])->getName()) << "\n";
        }
        free(page);
      }
  );

  // and now, simply run the pipeline and then destroy it!!!
  std::cout << "\nRUNNING PIPELINE\n";
  myPipeline->run();
  myPipeline = nullptr;

  // and be sure to delete the contents of the ComputePlan object... this always needs to be done
  // before the object is written to disk or sent accross the network, so that we don't end up
  // moving around a C++ smart pointer, which would be bad
  myPlan->nullifyPlanPointer();

}