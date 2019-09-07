#include <PDBClient.h>
#include <SharedEmployee.h>
#include <GenericWork.h>
#include <ScanSupervisorSet.h>
#include <SillyQuery.h>
#include <SillyAgg.h>
#include <FinalQuery.h>
#include <WriteSalaries.h>
#include <gtest/gtest.h>
#include <ReadInt.h>
#include <IntUnion.h>
#include <IntWriter.h>
#include <UnionIntSelection.h>
#include "ScanEmployeeSet.h"
#include "EmployeeBuiltInIdentitySelection.h"
#include "WriteBuiltinEmployeeSet.h"

using namespace pdb;

const size_t blockSize = 64;
const size_t numPages = 2;
int last = 0;

void fillSet(pdb::PDBClient &pdbClient, const std::string &db, const std::string &set) {

  //
  int start = 0;

  // make the allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

  // write a bunch of supervisors to it
  Handle<Vector<Handle<int>>> data = pdb::makeObject<Vector<Handle<int>>>();

  for(int i = 0; i < numPages; ++i) {
    try {

      // fill the vector up
      for (; true; i++) {
        Handle<int> myInt = makeObject<int>(start++);
        data->push_back(myInt);
      }

    } catch (pdb::NotEnoughSpace &e) {

      // remove the last int
      data->pop_back();

      last = std::max(last, *((*data)[data->size() - 1]));

      // send the data a bunch of times
      pdbClient.sendData<int>(db, set, data);
    }
  }

}

int main(int argc, char* argv[]) {

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libReadInt.so");
  pdbClient.registerType("libraries/libIntUnion.so");
  pdbClient.registerType("libraries/libIntWriter.so");
  pdbClient.registerType("libraries/libUnionIntSelection.so");

  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("chris_db");

  // now, create the input and output sets
  pdbClient.createSet<int>("chris_db", "input_set1");
  pdbClient.createSet<int>("chris_db", "input_set2");
  pdbClient.createSet<int>("chris_db", "input_set3");
  pdbClient.createSet<int>("chris_db", "output_set");

  /// 3. Fill in the data

  fillSet(pdbClient, "chris_db", "input_set1");
  fillSet(pdbClient, "chris_db", "input_set2");
  fillSet(pdbClient, "chris_db", "input_set3");

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  /// 5. create all of the computation objects and run the query

  // create all of the computation objects
  Handle<Computation> myScanSet1 = makeObject<ReadInt>("chris_db", "input_set1");
  Handle<UnionIntSelection> selection1 = makeObject<UnionIntSelection>(0);
  selection1->setInput(myScanSet1);

  Handle<Computation> myScanSet2 = makeObject<ReadInt>("chris_db", "input_set2");
  Handle<UnionIntSelection> selection2 = makeObject<UnionIntSelection>(1);
  selection2->setInput(myScanSet2);

  Handle<Computation> myScanSet3 = makeObject<ReadInt>("chris_db", "input_set3");
  Handle<UnionIntSelection> selection3 = makeObject<UnionIntSelection>(2);
  selection3->setInput(myScanSet3);

  Handle<Computation> myQuery1 = makeObject<IntUnion>();
  myQuery1->setInput(0, selection1);
  myQuery1->setInput(1, selection2);

  Handle<Computation> myQuery2 = makeObject<IntUnion>();
  myQuery2->setInput(0, myQuery1);
  myQuery2->setInput(1, selection3);

  Handle<Computation> myWriteSet = makeObject<IntWriter>("chris_db", "output_set");
  myWriteSet->setInput(myQuery2);

  // run computations
  pdbClient.executeComputations({ myWriteSet });

  /// 5. Get the set from the

  // grab the iterator
  auto it = pdbClient.getSetIterator<int>("chris_db", "output_set");

  std::vector<bool> checkArray((size_t) last);

  int i = 0;
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();

    // set the array to true
    checkArray[*r] = true;

    // go to the next one
    i++;
  }

  for(bool chk : checkArray) {
    if(!chk) {
      std::cout << "Did not get all the numbers" << '\n';
    }
  }

  if(last + 1 != i) {
    std::cout << "Some records are missing!";
  }

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}