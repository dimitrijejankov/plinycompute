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

  atomic_int count;
  count = 0;

  std::vector<std::string> names = {"Frank", "Joe", "Mark", "David", "Zoe"};
  for(int j = 0; j < 5; j++) {

      // allocate the thing
      pdb::makeObjectAllocatorBlock(blockSize * 1024l * 1024, true);

      // allocate the vector
      pdb::Handle<pdb::Vector<pdb::Handle<Employee>>> storeMe = pdb::makeObject<pdb::Vector<pdb::Handle<Employee>>>();

      try {

        for (int i = 0; true; i++) {

          pdb::Handle<Employee> myData;

          if (i % 100 == 0) {
            myData = pdb::makeObject<Employee>(names[j] + " Frank", count);
          } else {
            myData = pdb::makeObject<Employee>(names[j] + " " + to_string(count), count + 45);
          }

          storeMe->push_back(myData);
        }

      } catch (pdb::NotEnoughSpace &n) {

        storeMe->pop_back();
        count += storeMe->size();
        pdbClient.sendData<Employee>("chris_db", "chris_set", storeMe);
      }
  }

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  // here is the list of computations
  Handle<Computation> myScanSet = makeObject<ScanEmployeeSet>("chris_db", "chris_set");
  Handle<Computation> myQuery = makeObject<EmployeeBuiltInIdentitySelection>();
  myQuery->setInput(myScanSet);
  Handle<Computation> myWriteSet = makeObject<WriteBuiltinEmployeeSet>("chris_db", "output_set");
  myWriteSet->setInput(myQuery);

  // run computations
  pdbClient.executeComputations({ myWriteSet });

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