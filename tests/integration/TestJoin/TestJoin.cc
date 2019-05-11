#include <PDBClient.h>
#include <SharedEmployee.h>
#include <GenericWork.h>
#include <SillyReadOfA.h>
#include <SillyReadOfB.h>
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
size_t numToJoin = 0;

void fillSetAPageWithData(pdb::PDBClient &pdbClient) {

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
    numToJoin = std::min(numToJoin, i);

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
    numToJoin = std::min(numToJoin, i);

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
  pdbClient.registerType("libraries/libSillyReadOfA.so");
  pdbClient.registerType("libraries/libSillyReadOfB.so");
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

  fillSetAPageWithData(pdbClient);
  fillSetBPageWithData(pdbClient);

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  pdb::String tcapString = "A(a) <= SCAN ('myData', 'mySetA', 'SetScanner_0')\n"
                           "B(b) <= SCAN ('myData', 'mySetB', 'SetScanner_1')\n"
                           "A_extracted_value(a,self_0_2Extracted) <= APPLY (A(a), A(a), 'JoinComp_2', 'self_0', [('lambdaType', 'self')])\n"
                           "AHashed(a,a_value_for_hashed) <= HASHLEFT (A_extracted_value(self_0_2Extracted), A_extracted_value(a), 'JoinComp_2', '==_2', [])\n"
                           "B_extracted_value(b,b_value_for_hash) <= APPLY (B(b), B(b), 'JoinComp_2', 'attAccess_1', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                           "BHashedOnA(b,b_value_for_hashed) <= HASHRIGHT (B_extracted_value(b_value_for_hash), B_extracted_value(b), 'JoinComp_2', '==_2', [])\n"
                           "\n"
                           "/* Join ( a ) and ( b ) */\n"
                           "AandBJoined(a, b) <= JOIN (AHashed(a_value_for_hashed), AHashed(a), BHashedOnA(b_value_for_hashed), BHashedOnA(b), 'JoinComp_2')\n"
                           "AandBJoined_WithLHSExtracted(a,b,LHSExtractedFor_2_2) <= APPLY (AandBJoined(a), AandBJoined(a,b), 'JoinComp_2', 'self_0', [('lambdaType', 'self')])\n"
                           "AandBJoined_WithBOTHExtracted(a,b,LHSExtractedFor_2_2,RHSExtractedFor_2_2) <= APPLY (AandBJoined_WithLHSExtracted(b), AandBJoined_WithLHSExtracted(a,b,LHSExtractedFor_2_2), 'JoinComp_2', 'attAccess_1', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                           "AandBJoined_BOOL(a,b,bool_2_2) <= APPLY (AandBJoined_WithBOTHExtracted(LHSExtractedFor_2_2,RHSExtractedFor_2_2), AandBJoined_WithBOTHExtracted(a,b), 'JoinComp_2', '==_2', [('lambdaType', '==')])\n"
                           "AandBJoined_FILTERED(a, b) <= FILTER (AandBJoined_BOOL(bool_2_2), AandBJoined_BOOL(a, b), 'JoinComp_2')\n"
                           "\n"
                           "/* run Join projection on ( a b )*/\n"
                           "AandBJoined_Projection (nativ_3_2OutFor) <= APPLY (AandBJoined_FILTERED(a,b), AandBJoined_FILTERED(), 'JoinComp_2', 'native_lambda_3', [('lambdaType', 'native_lambda')])\n"
                           "out( ) <= OUTPUT ( AandBJoined_Projection ( nativ_3_2OutFor ), 'myData', 'outSet', 'SetWriter_3')";

  // here is the list of computations
  Handle<Vector<Handle<Computation>>> myComputations = makeObject<Vector<Handle<Computation>>>();

  // here is the list of computations
  Handle <Computation> readA = makeObject <SillyReadOfA>();
  Handle <Computation> readB = makeObject <SillyReadOfB>();
  Handle <Computation> join = makeObject <SillyJoinIntString>();
  join->setInput(0, readA);
  join->setInput(1, readB);
  Handle <Computation> write = makeObject <SillyWriteIntString>();
  write->setInput(0, join);

  // put them in the list of computations
  myComputations->push_back(readA);
  myComputations->push_back(readB);
  myComputations->push_back(join);
  myComputations->push_back(write);

  //TODO this is just a preliminary version of the execute computation before we add back the TCAP generation
  pdbClient.executeComputations(myComputations, tcapString);

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
      pdbClient.shutDownServer();
      exit(-1);
    }
  });

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}