#include <PDBClient.h>
#include <SharedEmployee.h>
#include <GenericWork.h>
#include <ScanSupervisorSet.h>
#include <SillyQuery.h>
#include <SillyAgg.h>
#include <FinalQuery.h>
#include <StringIntPair.h>
#include <WriteSalaries.h>
#include <ReadStringIntPair.h>
#include <StringIntPairMultiSelection.h>
#include <WriteStringIntPair.h>
#include "ScanEmployeeSet.h"
#include "EmployeeBuiltInIdentitySelection.h"
#include "WriteBuiltinEmployeeSet.h"

using namespace pdb;

const size_t blockSize = 64;
const size_t numToReplicate = 8;
size_t numToRetrieve = 0;

void fillSetPageWithData(pdb::PDBClient &pdbClient) {

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
    numToRetrieve = ((i / 10) + 1) * 10;

    // send the data a bunch of times
    for(size_t j = 0; j < numToReplicate; ++j) {
      pdbClient.sendData<StringIntPair>("chris_db", "chris_set", data);
    }
  }
}


int main(int argc, char* argv[]) {

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libReadStringIntPair.so");
  pdbClient.registerType("libraries/libStringIntPairMultiSelection.so");
  pdbClient.registerType("libraries/libWriteStringIntPair.so");

  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("chris_db");

  // now, create the input and output sets
  pdbClient.createSet<StringIntPair>("chris_db", "chris_set");
  pdbClient.createSet<StringIntPair>("chris_db", "output_set");

  /// 3. Fill in the data

  fillSetPageWithData(pdbClient);

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  /// 5. create all of the computation objects and run the query

  Handle <Computation> readStringIntPair = pdb::makeObject <ReadStringIntPair>("chris_db", "chris_set");
  Handle <Computation> multiSelection = pdb::makeObject <StringIntPairMultiSelection>();
  multiSelection->setInput(0, readStringIntPair);
  Handle <Computation> writeStringIntPair = pdb::makeObject <WriteStringIntPair>("chris_db", "output_set");
  writeStringIntPair->setInput(0, multiSelection);

  // run computations
  pdbClient.executeComputations({ writeStringIntPair });

  /// 6. Get the set from the

  std::unordered_map<int, int> counts;
  for(int i = 0; i < numToRetrieve; ++i) { counts[i] = numToReplicate; }

  // grab the iterator
  auto it = pdbClient.getSetIterator<StringIntPair>("chris_db", "output_set");
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();
    if(std::string("Hi") != r->myString->c_str()) {
      std::cout << r->myString->c_str() << " does not equal Hi" << std::endl;
      pdbClient.shutDownServer();
    }
    counts[r->myInt]--;
  }

  // make sure we had every record
  for_each (counts.begin(), counts.end(), [&](auto &count) {
    if(count.second != 0) {
      std::cout << "Not all records received  \n";
      pdbClient.shutDownServer();
    }
  });

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}