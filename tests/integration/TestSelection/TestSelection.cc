#include <PDBClient.h>
#include <SharedEmployee.h>
#include <GenericWork.h>
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
  pdbClient.registerType("libraries/libEmployeeBuiltInIdentitySelection.so");
  pdbClient.registerType("libraries/libWriteBuiltinEmployeeSet.so");
  pdbClient.registerType("libraries/libScanEmployeeSet.so");


  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("chris_db");

  // now, create the input and output sets
  pdbClient.createSet<Employee>("chris_db", "chris_set");
  pdbClient.createSet<Employee>("chris_db", "output_set");

  /// 3. Fill in the data multi-threaded

  // init the worker threads of this server
  auto workers = make_shared<pdb::PDBWorkerQueue>(make_shared<pdb::PDBLogger>("worker.log"),  10);

  // create the buzzer
  atomic_int counter;
  counter = 0;
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {
    cnt++;
  });

  atomic_int count;
  count = 0;

  std::vector<std::string> names = {"Frank", "Joe", "Mark", "David", "Zoe"};
  for(int j = 0; j < 5; j++) {

    // the thread
    int thread = j;

    // grab a worker
    pdb::PDBWorkerPtr myWorker = workers->getWorker();

    // start the thread
    pdb::PDBWorkPtr myWork = make_shared<pdb::GenericWork>([&, thread](PDBBuzzerPtr callerBuzzer) {

      // allocate the thing
      pdb::makeObjectAllocatorBlock(blockSize * 1024l * 1024, true);

      // allocate the vector
      pdb::Handle<pdb::Vector<pdb::Handle<Employee>>> storeMe = pdb::makeObject<pdb::Vector<pdb::Handle<Employee>>>();

      try {

        for (int i = 0; true; i++) {

          pdb::Handle<Employee> myData;

          if (i % 100 == 0) {
            myData = pdb::makeObject<Employee>(names[thread] + " Frank", count);
          } else {
            myData = pdb::makeObject<Employee>(names[thread] + " " + to_string(count), count + 45);
          }


          storeMe->push_back(myData);
          count++;
        }

      } catch (pdb::NotEnoughSpace &n) {

        pdbClient.sendData<Employee>("chris_db", "chris_set", storeMe);
      }

      // excellent everything worked just as expected
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    myWorker->execute(myWork, tempBuzzer);
  }

  // wait until all the nodes are finished
  while (counter < 5) {
    tempBuzzer->wait();
  }

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  String tcap = "inputDataForScanSet_0(in0) <= SCAN ('chris_db', 'chris_set', 'SetScanner_0') \n"\
                "nativ_0OutForSelectionComp1(in0,nativ_0_1OutFor) <= APPLY (inputDataForScanSet_0(in0), inputDataForScanSet_0(in0), 'SelectionComp_1', 'native_lambda_0', [('lambdaType', 'native_lambda')]) \n"\
                "filteredInputForSelectionComp1(in0) <= FILTER (nativ_0OutForSelectionComp1(nativ_0_1OutFor), nativ_0OutForSelectionComp1(in0), 'SelectionComp_1') \n"\
                "nativ_1OutForSelectionComp1 (nativ_1_1OutFor) <= APPLY (filteredInputForSelectionComp1(in0), filteredInputForSelectionComp1(), 'SelectionComp_1', 'native_lambda_1', [('lambdaType', 'native_lambda')]) \n"\
                "nativ_1OutForSelectionComp1_out() <= OUTPUT ( nativ_1OutForSelectionComp1 ( nativ_1_1OutFor ), 'chris_db', 'output_set', 'SetWriter_2') \n";

  // here is the list of computations
  Handle<Vector<Handle<Computation>>> myComputations = makeObject<Vector<Handle<Computation>>>();

  // here is the list of computations
  Handle<Computation> myScanSet = makeObject<ScanEmployeeSet>();
  Handle<Computation> myQuery = makeObject<EmployeeBuiltInIdentitySelection>();
  myQuery->setInput(myScanSet);
  Handle<Computation> myWriteSet = makeObject<WriteBuiltinEmployeeSet>("chris_db", "chris_set");
  myWriteSet->setInput(myQuery);

  // put them in the list of computations
  myComputations->push_back(myScanSet);
  myComputations->push_back(myQuery);
  myComputations->push_back(myWriteSet);

  //TODO this is just a preliminary version of the execute computation before we add back the TCAP generation
  pdbClient.executeComputations(myComputations, tcap);

  /// 5. Get the set from the

  // grab the iterator
  auto it = pdbClient.getSetIterator<Employee>("chris_db", "output_set");

  int i = 0;
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();

    // print every 100th
    if(i % 100 == 0) {
      std::cout << *r->getName() << std::endl;
    }

    // go to the next one
    i++;
  }

  std::cout << "Got " << i << " : " << "Stored " << count << std::endl;

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}