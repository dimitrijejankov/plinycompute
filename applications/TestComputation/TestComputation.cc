#include <PDBClient.h>
#include <GenericWork.h>
#include <fstream>
#include "TRABlock.h"
#include "sharedLibraries/headers/MatrixScanner.h"
#include "sharedLibraries/headers/MatrixMultiplyJoin.h"
#include "sharedLibraries/headers/MatrixSumJoin.h"
#include "sharedLibraries/headers/MatrixMultiplyAggregation.h"
#include "sharedLibraries/headers/MatrixWriter.h"

using namespace pdb;


const size_t blockSize = 128;
const uint32_t matrixRows = 40000;
const uint32_t matrixColumns = 40000;
const uint32_t numRows = 40;
const uint32_t numCols = 40;

void initMatrix(pdb::PDBClient &pdbClient, const std::string &set) {

  // fill the vector up
  std::vector<std::pair<uint32_t, uint32_t>> tuplesToSend;
  for (uint32_t r = 0; r < numRows; r++) {
    for (uint32_t c = 0; c < numCols; c++) {
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
      int n = 4;
      for(; i < tuplesToSend.size(); ++i) {

        n--;
        if(n == 0) {
          break;
        }
        // allocate a matrix
        Handle<TRABlock> myInt = makeObject<TRABlock>(tuplesToSend[i].first,
                                                            tuplesToSend[i].second,
                                                            matrixRows / numRows,
                                                            matrixColumns / numCols);

        // init the values
        float *vals = myInt->data->data->c_ptr();
        for (int v = 0; v < (matrixRows / numRows) * (matrixColumns / numCols); ++v) {
          vals[v] = 1.0f;
        }

        // we add the matrix to the block
        data->push_back(myInt);


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

int main(int argc, char *argv[]) {

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libMatrixBlock.so");
  pdbClient.registerType("libraries/libMatrixBlockData.so");
  pdbClient.registerType("libraries/libMatrixBlockMeta.so");
  pdbClient.registerType("libraries/libMatrixMultiplyAggregation.so");
  pdbClient.registerType("libraries/libMatrixMultiplyJoin.so");
  pdbClient.registerType("libraries/libMatrixSumJoin.so");
  pdbClient.registerType("libraries/libMatrixScanner.so");
  pdbClient.registerType("libraries/libMatrixWriter.so");


  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("myData");

  // now, create the input and output sets
  pdbClient.createSet<TRABlock>("myData", "A");
  pdbClient.createSet<TRABlock>("myData", "B");
  pdbClient.createSet<TRABlock>("myData", "C");
  pdbClient.createSet<TRABlock>("myData", "D");

  /// 3. Fill in the data (single threaded)

  initMatrix(pdbClient, "A");
  initMatrix(pdbClient, "B");
  initMatrix(pdbClient, "C");

  pdbClient.createIndex("myData", "A");
  pdbClient.createIndex("myData", "B");
  pdbClient.createIndex("myData", "C");

  /// 4. Make query graph an run query

  // for allocations
  const pdb::UseTemporaryAllocationBlock tempBlock{128 * 1024 * 1024};

  // create all of the computation objects
  Handle<Computation> a1 = pdb::makeObject<pdb::matrix::MatrixScanner>("myData", "A");
  Handle<Computation> a2 = pdb::makeObject<pdb::matrix::MatrixScanner>("myData", "B");
  Handle<Computation> a3 = pdb::makeObject<pdb::matrix::MatrixScanner>("myData", "C");

  // make the joinAdd
  Handle<Computation> joinAdd = pdb::makeObject<pdb::matrix::MatrixSumJoin>();
  joinAdd->setInput(0, a1);
  joinAdd->setInput(1, a2);

  // create the multiply
  Handle<Computation> joinMul = pdb::makeObject<pdb::matrix::MatrixSumJoin>();
  joinMul->setInput(0, joinAdd);
  joinMul->setInput(1, a3);

  // make the aggregation
  Handle<Computation> myAggregation = makeObject<pdb::matrix::MatrixMultiplyAggregation>();
  myAggregation->setInput(joinMul);

  // make the writer
  Handle<Computation> writeStringIntPair = pdb::makeObject<pdb::matrix::MatrixWriter>("myData", "D");
  writeStringIntPair->setInput(0, myAggregation);

  std::chrono::steady_clock::time_point planner_begin = std::chrono::steady_clock::now();

  //TODO this is just a preliminary version of the execute computation before we add back the TCAP generation
  bool success = pdbClient.executeComputations({writeStringIntPair});

  std::chrono::steady_clock::time_point planner_end = std::chrono::steady_clock::now();
  std::cout << "Computation run for "
            << (double) std::chrono::duration_cast<std::chrono::nanoseconds>(planner_end - planner_begin).count() / (double) std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count()
            << "[s]" << '\n';


  /// 5. Get the set from the

  // grab the iterator
  int count = 0;
  auto it = pdbClient.getSetIterator<TRABlock>("myData", "D");
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();

    //r->print();

    count++;
  }

  std::cout << count << "\n";

  // wait a bit before the shutdown
  sleep(4);

  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}