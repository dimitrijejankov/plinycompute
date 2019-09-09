#include <PDBClient.h>
#include <SharedEmployee.h>
#include <GenericWork.h>
#include <ReadInt.h>
#include <ReadStringIntPair.h>
#include <SillyCartasianJoinIntString.h>
#include <SillyWriteIntString.h>
#include "StringIntPair.h"
#include "ScanEmployeeSet.h"
#include "EmployeeBuiltInIdentitySelection.h"
#include "WriteBuiltinEmployeeSet.h"

using namespace pdb;

// some constants for the test
const size_t blockSize = 1;

// the number of keys that are going to be joined
size_t numA = 0;
size_t numB = 0;

void fillSet(pdb::PDBClient &pdbClient) {

  // make the allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024};

  // write a bunch of supervisors to it
  Handle<Vector<Handle<int>>> data = pdb::makeObject<Vector<Handle<int>>>();

  // fill the vector up
  size_t i = 0;
  for (; true; i++) {
    Handle<int> myInt = makeObject<int>(i);
    data->push_back(myInt);

    if(i > 10) {
      break;
    }
  }

  // how many did we have
  numA = data->size() * 2;

  // send the data a once
  pdbClient.sendData<int>("myData", "mySetA", data);
  pdbClient.sendData<int>("myData", "mySetA", data);
}

void fillSetBPageWithData(pdb::PDBClient &pdbClient) {

  // make the allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

  // write a bunch of supervisors to it
  Handle <Vector <Handle <StringIntPair>>> data = pdb::makeObject<Vector <Handle <StringIntPair>>>();

  // fill the vector up
  size_t i = 0;
  for (; true; i++) {
    std::ostringstream oss;
    oss << "My string is " << i;
    oss.str();
    Handle <StringIntPair> myPair = makeObject <StringIntPair> (oss.str (), i);
    data->push_back (myPair);

    if(i > 800) {
      break;
    }
  }

  // how many did we have
  numB = data->size() * 2;

  // send the data once
  pdbClient.sendData<StringIntPair>("myData", "mySetB", data);
  pdbClient.sendData<StringIntPair>("myData", "mySetB", data);
}

int main(int argc, char* argv[]) {

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libReadInt.so");
  pdbClient.registerType("libraries/libReadStringIntPair.so");
  pdbClient.registerType("libraries/libSillyCartasianJoinIntString.so");
  pdbClient.registerType("libraries/libSillyWriteIntString.so");

  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("myData");

  // now, create the input and output sets
  pdbClient.createSet<int>("myData", "mySetA");
  pdbClient.createSet<StringIntPair>("myData", "mySetB");
  pdbClient.createSet<String>("myData", "outSet");

  /// 3. Fill in the data (single threaded)

  fillSet(pdbClient);
  fillSetBPageWithData(pdbClient);

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  // here is the list of computations
  Handle<Vector<Handle<Computation>>> myComputations = makeObject<Vector<Handle<Computation>>>();

  // here is the list of computations
  Handle <Computation> readA = makeObject <ReadInt>("myData", "mySetA");
  Handle <Computation> readB = makeObject <ReadStringIntPair>("myData", "mySetB");
  Handle <Computation> join = makeObject <SillyCartasianJoinIntString>();
  join->setInput(0, readA);
  join->setInput(1, readB);
  Handle <Computation> write = makeObject <SillyWriteIntString>("myData", "outSet");
  write->setInput(0, join);

  // put them in the list of computations
  myComputations->push_back(readA);
  myComputations->push_back(readB);
  myComputations->push_back(join);
  myComputations->push_back(write);

  //TODO this is just a preliminary version of the execute computation before we add back the TCAP generation
  pdbClient.executeComputations({ write });

  /// 5. Get the set from the

  size_t count = 0;

  // grab the iterator
  auto it = pdbClient.getSetIterator<String>("myData", "outSet");
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();

    // count the stuff
    count++;
  }

  // check the number returned
  if(count == numA * numB) {
    std::cout << "got correct number back " << count << " " << numA * numB;
  }
  else {
    std::cout << "got wrong number back " << count << " " << numA * numB;
  }

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}