#include <PDBClient.h>
#include <GenericWork.h>
#include <random>
#include "TRABlock.h"
#include "sharedLibraries/headers/TensorScanner.h"
#include "sharedLibraries/headers/BCMMJoin.h"
#include "sharedLibraries/headers/BCMMAggregation.h"
#include "sharedLibraries/headers/TensorWriter.h"
#include "sharedLibraries/headers/MatrixSumJoin.h"

using namespace pdb;
using namespace pdb::matrix;

// some constants for the test
const size_t blockSize = 64;

const uint32_t n = 5000;
const uint32_t n_block = 1000;

const bool doNotPrint = true;

void initMatrix(pdb::PDBClient &pdbClient, const std::string &set,
                int rows, int cols, int row_block, int col_block) {

  // figure out the number of blocks
  int numRowsBlocks = rows / row_block;
  int numColsBlocks = cols / col_block;

  // fill the vector up
  std::vector<std::pair<uint32_t, uint32_t>> tuplesToSend;
  for (uint32_t r = 0; r < numRowsBlocks; r++) {
    for (uint32_t c = 0; c < numColsBlocks; c++) {
      tuplesToSend.emplace_back(std::make_pair(r, c));
    }
  }

  // make the allocation block
  size_t i = 0;
  while(i != tuplesToSend.size()) {

    // use temporary allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

    // put the chunks here
    Handle<Vector<Handle<TRABlock>>> data = pdb::makeObject<Vector<Handle<TRABlock>>>();

    try {

      // put stuff into the vector
      for(; i < tuplesToSend.size();) {

        // allocate a matrix
        Handle<TRABlock> myInt = makeObject<TRABlock>(tuplesToSend[i].first, tuplesToSend[i].second, 0,
                                                      row_block, col_block, 1);

        // init the values
        float *vals = myInt->data->data->c_ptr();
        for (int v = 0; v < row_block * col_block; ++v) {
          vals[v] = 1.0f;
        }

        // we add the matrix to the block
        data->push_back(myInt);

        // go to the next one
        ++i;

        if(data->size() == 8) {
          break;
        }
      }
    }
    catch (pdb::NotEnoughSpace &n) {}

    // init the records
    getRecord(data);

    // send the data a bunch of times
    pdbClient.sendData<TRABlock>("myData", set, data);

    // log that we stored stuff
    std::cout << "Stored " << data->size() << " !\n";
  }

}

int main(int argc, char* argv[]) {

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libBCMMAggregation.so");
  pdbClient.registerType("libraries/libBCMMJoin.so");
  pdbClient.registerType("libraries/libMatrixSumJoin.so");
  pdbClient.registerType("libraries/libTensorScanner.so");
  pdbClient.registerType("libraries/libTensorWriter.so");

  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("myData");

  // now, create the input and output sets
  pdbClient.createSet<TRABlock>("myData", "A");
  pdbClient.createSet<TRABlock>("myData", "B");
  pdbClient.createSet<TRABlock>("myData", "C");
  pdbClient.createSet<TRABlock>("myData", "D");

  /// 3. Fill in the data (single threaded)

  initMatrix(pdbClient, "A", n, n, n_block, n_block);
  initMatrix(pdbClient, "B", n, n, n_block, n_block);
  initMatrix(pdbClient, "C", n, n, n_block, n_block);

  // you have to do this to reference a page set in the TRA interface
  pdbClient.createIndex("myData", "A");
  pdbClient.createIndex("myData", "B");
  pdbClient.createIndex("myData", "C");

  /// 4. Make query graph an run query
  std::chrono::steady_clock::time_point planner_begin = std::chrono::steady_clock::now();
  // do the sum
  {
    // for allocations
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    Handle <Computation> readA = makeObject <TensorScanner>("myData", "A");
    Handle <Computation> readB = makeObject <TensorScanner>("myData", "B");
    Handle <Computation> join = makeObject <MatrixSumJoin>();
    join->setInput(0, readA);
    join->setInput(1, readB);
    Handle<Computation> myAggregation = makeObject<BCMMAggregation>();
    myAggregation->setInput(join);
    Handle<Computation> myWriter = makeObject<TensorWriter>("myData", "C");
    myWriter->setInput(myAggregation);

    pdbClient.shuffle("myData:A", {1}, "AShuffled");
    pdbClient.shuffle("myData:B", {1}, "BShuffled");
    pdbClient.localJoin("AShuffled", {0, 1}, "BShuffled", {0, 1}, { myWriter }, "ABJoined",
                        "OutForJoinedFor_equals_0JoinComp2", "OutFor_joinRec_5JoinComp2");
  }

  // do the multiply
  {
    // for allocations
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    Handle <Computation> readA = makeObject <TensorScanner>("myData", "A");
    Handle <Computation> readB = makeObject <TensorScanner>("myData", "B");
    Handle <Computation> join = makeObject <BCMMJoin>();
    join->setInput(0, readA);
    join->setInput(1, readB);
    Handle<Computation> myAggregation = makeObject<BCMMAggregation>();
    myAggregation->setInput(join);
    Handle<Computation> myWriter = makeObject<TensorWriter>("myData", "C");
    myWriter->setInput(myAggregation);

    pdbClient.shuffle("myData:C", {0}, "CShuffled");
    pdbClient.localJoin("ABJoined", {1}, "CShuffled", {0}, { myWriter }, "ABCJoined",
                        "OutForJoinedFor_equals_0JoinComp2", "OutFor_joinRec_5JoinComp2");
    pdbClient.shuffle("ABCJoined", {0, 1}, "ABCJoinedShuffled");
    pdbClient.localAggregation("ABCJoinedShuffled", {0, 1},  "Final");
  }

  pdbClient.materialize("myData", "D", "Final");

  std::chrono::steady_clock::time_point planner_end = std::chrono::steady_clock::now();
  std::cout << "Computation run for "
            << (double) std::chrono::duration_cast<std::chrono::nanoseconds>(planner_end - planner_begin).count() / (double) std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count()
            << "[s]" << '\n';

  // grab the iterator
  auto it = pdbClient.getSetIterator<TRABlock>("myData", "D");
  int32_t count = 0;
  while (it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();
    //r->print();
    count++;
  }

  std::cout << "Count " << count << '\n';

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}