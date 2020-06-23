#include <PDBClient.h>
#include <GenericWork.h>
#include <random>
#include "sharedLibraries/headers/BPStrip.h"
#include "sharedLibraries/headers/BPScanner.h"
#include "sharedLibraries/headers/BPMultiplyJoin.h"
#include "sharedLibraries/headers/BPMultiplyAggregation.h"
#include "sharedLibraries/headers/BPWriter.h"

using namespace pdb;
using namespace pdb::bp;

// some constants for the test
const size_t blockSize = 64;
const size_t numBatches = 3;
const size_t batchSize = 11;
const size_t hiddenSize = 10;
const size_t outputSize = 12;

const bool doNotPrint = false;

void initBP(pdb::PDBClient &pdbClient, int32_t rows, int32_t columns, const std::string &set) {

  // make the allocation block
  int32_t i = 0;
  while(i < numBatches) {

    // use temporary allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

    // put the chunks here
    Handle<Vector<Handle<BPStrip>>> data = pdb::makeObject<Vector<Handle<BPStrip>>>();

    try {

      // put stuff into the vector
      for(; i < numBatches;) {

        // allocate a matrix
        Handle<BPStrip> myBlock = makeObject<BPStrip>(i, rows, columns);

        // init the values
        float *vals = myBlock->data->data->c_ptr();
        for (int v = 0; v < rows * columns; ++v) {
          vals[v] = 1.0f * i;
        }

        // we add the matrix to the block
        data->push_back(myBlock);

        // go to the next one
        ++i;

        if(data->size() == 50) {
          break;
        }
      }
    }
    catch (pdb::NotEnoughSpace &n) {}

    // init the records
    getRecord(data);

    // send the data a bunch of times
    pdbClient.sendData<BPStrip>("myData", set, data);

    // log that we stored stuff
    std::cout << "Stored " << data->size() << " !\n";
  }
}

int main(int argc, char* argv[]) {

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libBPStrip.so");
  pdbClient.registerType("libraries/libBPStripData.so");
  pdbClient.registerType("libraries/libBPStripMeta.so");
  pdbClient.registerType("libraries/libBPMultiplyAggregation.so");
  pdbClient.registerType("libraries/libBPMultiplyJoin.so");
  pdbClient.registerType("libraries/libBPScanner.so");
  pdbClient.registerType("libraries/libBPWriter.so");

  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("myData");

  // now, create the input and output sets
  pdbClient.createSet<BPStrip>("myData", "A");
  pdbClient.createSet<BPStrip>("myData", "B");
  pdbClient.createSet<BPStrip>("myData", "C");

  /// 3. Fill in the data (single threaded)

  initBP(pdbClient, batchSize, hiddenSize, "A");
  initBP(pdbClient, batchSize, outputSize, "B");

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  Handle <Computation> readA = makeObject <BPScanner>("myData", "A");
  Handle <Computation> readB = makeObject <BPScanner>("myData", "B");
  Handle <Computation> join = makeObject <BPMultiplyJoin>();
  join->setInput(0, readA);
  join->setInput(1, readB);
  Handle<Computation> myAggregation = makeObject<BPMultiplyAggregation>();
  myAggregation->setInput(join);
  Handle<Computation> myWriter = makeObject<BPWriter>("myData", "C");
  myWriter->setInput(myAggregation);

  std::chrono::steady_clock::time_point planner_begin = std::chrono::steady_clock::now();

  //TODO this is just a preliminary version of the execute computation before we add back the TCAP generation
  bool success = pdbClient.executeComputations({ myWriter });

  std::chrono::steady_clock::time_point planner_end = std::chrono::steady_clock::now();
  std::cout << "Run multiply for " << std::chrono::duration_cast<std::chrono::nanoseconds>(planner_end - planner_begin).count()
            << "[ns]" << '\n';


  /// 5. Get the set from the

  // grab the iterator
  auto it = pdbClient.getSetIterator<BPStrip>("myData", "C");
  int32_t count = 0;
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();
    count++;

    // skip if we do not need to print
    if(doNotPrint) {
      continue;
    }

    // write out the values
    float *values = r->data->data->c_ptr();
    for(int i = 0; i < r->data->numRows; ++i) {
      for(int j = 0; j < r->data->numCols; ++j) {
        std::cout << values[i * r->data->numCols + j] << ", ";
      }
      std::cout << "\n";
    }

    std::cout << "\n\n";
  }

  // wait a bit before the shutdown
  sleep(4);

  std::cout << count << '\n';

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}