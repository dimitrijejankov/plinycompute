#include <PDBClient.h>
#include <SharedEmployee.h>
#include <GenericWork.h>
#include <ReadInt.h>
#include <ReadStringIntPair.h>
#include <SillyJoinIntString.h>
#include <SillyWriteIntString.h>
#include "StringIntPair.h"
#include "ScanEmployeeSet.h"
#include "EmployeeBuiltInIdentitySelection.h"
#include "WriteBuiltinEmployeeSet.h"

using namespace pdb;

// some constants for the test
const size_t blockSize = 64;
const size_t replicateCountForA = 3;
const size_t replicateCountForB = 2;

// the number of keys that are going to be joined
size_t numToJoin = std::numeric_limits<size_t>::max();

void fillSet(pdb::PDBClient &pdbClient) {

  // make the allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

  // write a bunch of supervisors to it
  Handle<Vector<Handle<int>>> data = pdb::makeObject<Vector<Handle<int>>>();

  size_t i = 0;
  try {

    // fill the vector up
    for (; true; i++) {
      Handle<int> myInt = makeObject<int>(i);
      data->push_back(myInt);
    }

  } catch (pdb::NotEnoughSpace &e) {

    // remove the last int
    data->pop_back();

    // how many did we have
    numToJoin = std::min(numToJoin, i - 1);

    // send the data a bunch of times
    for(size_t j = 0; j < replicateCountForA; ++j) {
      pdbClient.sendData<int>("myData", "mySetA", data);
    }
  }
}

void fillSetBPageWithData(pdb::PDBClient &pdbClient) {

  // make the allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

  // write a bunch of supervisors to it
  Handle <Vector <Handle <StringIntPair>>> data = pdb::makeObject<Vector <Handle <StringIntPair>>>();

  size_t i = 0;
  try {

    // fill the vector up
    for (; true; i++) {
      std::ostringstream oss;
      oss << "My string is " << i;
      oss.str();
      Handle <StringIntPair> myPair = makeObject <StringIntPair> (oss.str (), i);
      data->push_back (myPair);
    }

  } catch (pdb::NotEnoughSpace &e) {

    // remove the last string int pair
    data->pop_back();

    // how many did we have
    numToJoin = std::min(numToJoin, i - 1);

    // send the data a bunch of times
    for(size_t j = 0; j < replicateCountForB; ++j) {
      pdbClient.sendData<StringIntPair>("myData", "mySetB", data);
    }
  }
}

int main(int argc, char* argv[]) {

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libReadInt.so");
  pdbClient.registerType("libraries/libReadStringIntPair.so");
  pdbClient.registerType("libraries/libSillyJoinIntString.so");
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
  Handle <Computation> readA = makeObject <ReadInt>("myData", "mySetA");
  Handle <Computation> readB = makeObject <ReadStringIntPair>("myData", "mySetB");
  Handle <Computation> join = makeObject <SillyJoinIntString>();
  join->setInput(0, readA);
  join->setInput(1, readB);
  Handle <Computation> write = makeObject <SillyWriteIntString>("myData", "outSet");
  write->setInput(0, join);

  // run computations
  pdbClient.executeComputations({ write });

  /// 5. Get the set from the

  // grab the iterator
  auto it = pdbClient.getSetIterator<String>("myData", "outSet");

  std::unordered_map<int, int> counts;
  for(int i = 0; i < numToJoin; ++i) { counts[i] = replicateCountForA * replicateCountForB;}

  int i = 0;
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();

    // extract N from "Got int N and StringIntPair (N, 'My string is N')'";
    std::string tmp = r->c_str() + 8;
    std::size_t found = tmp.find(' ');
    tmp.resize(found);
    int n = std::stoi(tmp);

    // check the string
    std::string check = "Got int " + std::to_string(n) + " and StringIntPair ("  + std::to_string(n)  + ", '" + "My string is " + std::to_string(n) + "')'";
    if(check != r->c_str()) {
      std::cerr << "The string we got is not correct we wanted : " << std::endl;
      std::cerr << check << std::endl;
      std::cerr << "But got : " << std::endl;
      std::cerr << tmp << std::endl;

      // shutdown the server and exit
      pdbClient.shutDownServer();
      exit(-1);
    }

    // every join result must have an N less than numToJoin, since that is the common number keys to join
    if(n >= numToJoin) {
      std::cerr << r->c_str() << std::endl;
      std::cerr << "This is bad the key should always be less than numToJoin" << std::endl;

      // shutdown the server and exit
      pdbClient.shutDownServer();
      exit(-1);
    }

    counts[n]--;

    // go to the next one
    i++;
  }

  // make sure we had every record
  for_each (counts.begin(), counts.end(), [&](auto &count) {
    if(count.second != 0) {
      std::cerr << "Did not get the right count of records" << std::endl;
      std::cout << count.first << ", " << count.second << std::endl;
      //pdbClient.shutDownServer();
      //exit(-1);
    }
  });

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}